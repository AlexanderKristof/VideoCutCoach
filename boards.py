"""
A taktikai tábla mentései a szerveren (clips/boards/<id>.json), hogy az edző táblái
bármelyik eszközről elérhetők legyenek, amelyik a program szerverét használja.

Két fajta mentés van (kind):
  - "board":  felállás + rajzok (a tábla "Felállás és rajz" módja)
  - "replay": a visszajátszás rajz-megállói, a meccsfájl elérési útjával/URL-jével együtt
  - "video":  a videóra rajzolt rajz-megállók ({t, items, hold_sec, note}), a videó
              URL-jével / fájlnevével (video_url, video_name) – a "Videó elemzés" mód

Fájlforma: {"id", "name", "kind", "created_at", "updated_at", "data": {...}}

Csak a standard könyvtárat használja – fastapi nélkül tesztelhető (tests/test_boards.py).
A végpontok a main.py-ban vannak (GET/POST/DELETE /boards).
"""
import json
import os
import re
import threading
import time
import uuid

MAX_BOARD_BYTES = 2 * 1024 * 1024   # egy mentés legfeljebb 2 MB (JSON-ként)
MAX_NAME_LEN = 120
MAX_BOARDS = 2000                   # ennyi mentés fölött nem fogadunk újat
KINDS = ("board", "replay", "video")
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


class BoardError(Exception):
    """Felhasználónak szóló (magyar) hibaüzenet + HTTP státuszkód."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def clean_name(name):
    """Vezérlőkarakterek nélkül, szóközök összevonva, legfeljebb MAX_NAME_LEN karakter."""
    text = _CTRL_RE.sub(" ", str(name or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_NAME_LEN] or "Névtelen tábla"


def slugify(text):
    """Fájlnévnek is jó azonosító-előtag (ékezetek nélkül)."""
    table = str.maketrans("áéíóöőúüűÁÉÍÓÖŐÚÜŰ", "aeiooouuuAEIOOOUUU")
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(text or "").translate(table)).strip("-").lower()
    return s[:40] or "tabla"


def check_id(board_id):
    """Csak betű, szám, kötőjel, aláhúzás (max. 64) – így nem lehet a mappán kívülre írni."""
    if not isinstance(board_id, str) or not _ID_RE.match(board_id):
        raise BoardError("Érvénytelen tábla-azonosító.", 400)
    return board_id


def guess_kind(data):
    """'video', ha videós rajzsor; 'replay', ha rajz-megállókat tartalmaz; különben 'board'."""
    if isinstance(data, dict) and isinstance(data.get("stops"), list):
        return "video" if data.get("kind") == "video" else "replay"
    return "board"


def validate_data(data, kind):
    if not isinstance(data, dict):
        raise BoardError("A tábla adata (data) JSON objektum legyen.", 400)
    if kind == "board":
        board = data.get("board", data)
        if not isinstance(board, dict) or not isinstance(board.get("players", []), list) \
                or not isinstance(board.get("drawings", []), list):
            raise BoardError("A tábla adata hibás: a 'players' és a 'drawings' lista legyen.", 400)
    else:
        stops = data.get("stops")
        if not isinstance(stops, list):
            raise BoardError("A rajz-megállók mentésében a 'stops' lista legyen.", 400)
        for st in stops:
            if not isinstance(st, dict) or not isinstance(st.get("items", []), list):
                raise BoardError("Hibás rajz-megálló: mindegyiknek 't' ideje és 'items' listája legyen.", 400)
            try:
                float(st.get("t"))
            except (TypeError, ValueError):
                raise BoardError("Hibás rajz-megálló: a 't' idő szám legyen.", 400)


class BoardStore:
    """JSON-fájlos tároló egy mappában. Szálbiztos (egy folyamaton belül)."""

    def __init__(self, root, max_bytes=MAX_BOARD_BYTES, max_boards=MAX_BOARDS):
        self.root = os.path.abspath(root)
        self.max_bytes = int(max_bytes)
        self.max_boards = int(max_boards)
        self._lock = threading.Lock()
        os.makedirs(self.root, exist_ok=True)

    def _path(self, board_id):
        return os.path.join(self.root, check_id(board_id) + ".json")

    def _read(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list(self, kind=None):
        """[{id, name, kind, updated_at, created_at, size}], a legutóbb módosított elöl."""
        out = []
        for fn in os.listdir(self.root):
            if not fn.endswith(".json") or not _ID_RE.match(fn[:-5]):
                continue
            path = os.path.join(self.root, fn)
            try:
                doc = self._read(path)
            except (OSError, ValueError):
                continue  # sérült / félig írt fájl: kihagyjuk
            if not isinstance(doc, dict):
                continue
            if kind and doc.get("kind") != kind:
                continue
            data = doc.get("data") if isinstance(doc.get("data"), dict) else {}
            out.append({"id": fn[:-5], "name": doc.get("name") or fn[:-5], "kind": doc.get("kind", "board"),
                        "updated_at": doc.get("updated_at"), "created_at": doc.get("created_at"),
                        "size": os.path.getsize(path),
                        "events_url": data.get("events_url"),
                        "video_url": data.get("video_url") if doc.get("kind") == "video" else None,
                        "video_name": data.get("video_name") if doc.get("kind") == "video" else None})
        out.sort(key=lambda d: (d.get("updated_at") or "", d["id"]), reverse=True)
        return out

    def get(self, board_id):
        path = self._path(board_id)
        if not os.path.exists(path):
            raise BoardError("Nincs ilyen mentett tábla.", 404)
        try:
            return self._read(path)
        except (OSError, ValueError):
            raise BoardError("A mentett tábla fájlja sérült, nem olvasható.", 500)

    def save(self, name, data, board_id=None, kind=None):
        """
        Új mentés (board_id nélkül) vagy meglévő felülírása. Visszatér: a mentés meta-adatai
        (id, name, kind, created_at, updated_at, size).
        """
        if kind is None or kind == "":
            kind = guess_kind(data)
        if kind not in KINDS:
            raise BoardError("Ismeretlen mentésfajta (board, replay vagy video lehet).", 400)
        validate_data(data, kind)
        name = clean_name(name)
        if board_id:
            check_id(board_id)
        with self._lock:
            created = None
            if board_id and os.path.exists(self._path(board_id)):
                try:
                    created = self._read(self._path(board_id)).get("created_at")
                except (OSError, ValueError, AttributeError):
                    created = None
            else:
                count = sum(1 for fn in os.listdir(self.root) if fn.endswith(".json"))
                if count >= self.max_boards:
                    raise BoardError("Túl sok mentett tábla van a szerveren – törölj néhányat.", 507)
                board_id = board_id or f"{slugify(name)}-{uuid.uuid4().hex[:8]}"
            now = _now_iso()
            doc = {"id": board_id, "name": name, "kind": kind, "created_at": created or now,
                   "updated_at": now, "data": data}
            text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
            size = len(text.encode("utf-8"))
            if size > self.max_bytes:
                raise BoardError(f"A mentés túl nagy ({size / 1048576:.1f} MB, legfeljebb "
                                 f"{self.max_bytes / 1048576:.0f} MB).", 413)
            path = self._path(board_id)
            tmp = f"{path}.{uuid.uuid4().hex[:6]}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)  # atomikus csere: félig írt fájl nem marad
        return {"id": board_id, "name": name, "kind": kind, "created_at": doc["created_at"],
                "updated_at": now, "size": size}

    def delete(self, board_id):
        path = self._path(board_id)
        with self._lock:
            if not os.path.exists(path):
                raise BoardError("Nincs ilyen mentett tábla.", 404)
            os.remove(path)
        return True


def parse_request_body(raw_bytes, max_bytes=MAX_BOARD_BYTES):
    """
    A POST /boards törzse (bájtok) → (name, data, id, kind). A méretkorlátot a JSON
    feldolgozása előtt ellenőrzi; a hibák BoardError-ként jönnek.
    """
    if raw_bytes is None or len(raw_bytes) == 0:
        raise BoardError("Üres kérés: a 'name' és a 'data' mező kell.", 400)
    # a burkoló (id, name, kind) miatt egy kis ráhagyás
    if len(raw_bytes) > max_bytes + 4096:
        raise BoardError(f"A mentés túl nagy (legfeljebb {max_bytes / 1048576:.0f} MB).", 413)
    try:
        body = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise BoardError("A kérés törzse nem érvényes JSON.", 400)
    if not isinstance(body, dict):
        raise BoardError("A kérés törzse JSON objektum legyen.", 400)
    if "data" not in body:
        raise BoardError("Hiányzik a 'data' mező (a tábla vagy a rajz-megállók).", 400)
    board_id = body.get("id") or None
    if board_id is not None and not isinstance(board_id, str):
        raise BoardError("Érvénytelen tábla-azonosító.", 400)
    return body.get("name"), body.get("data"), board_id, body.get("kind")
