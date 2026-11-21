"""
A szerveres tábla-tároló (boards.py) tesztjei. Futtatás: python3 tests/test_boards.py
(pytest és fastapi nélkül fut)
"""
import os
import sys
import json
import shutil
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import boards
from boards import BoardStore, BoardError, parse_request_body

BOARD = {"board": {"form1": "4-3-3", "players": [{"id": "1-0", "team": 1, "x": 4, "y": 34}],
                   "drawings": [{"type": "arrow", "pts": [[1, 2], [3, 4]], "color": "#fff"}]}}
REPLAY = {"events_url": "/clips/match_events_ab12.json", "events_name": "match_events_ab12.json",
          "stops": [{"t": 3712.5, "items": [{"type": "circle", "pts": [[50, 30], [53, 30]], "color": "#ff4d4d"}]}]}


def _store(**kw):
    d = tempfile.mkdtemp(prefix="boards_test_")
    return d, BoardStore(os.path.join(d, "boards"), **kw)


def _raises(fn, status):
    try:
        fn()
    except BoardError as ex:
        assert ex.status == status, (ex.status, ex.message)
        assert ex.message  # magyar üzenet
        return ex
    raise AssertionError("BoardError várt")


def test_save_list_get_update_delete():
    d, st = _store()
    try:
        meta = st.save("Szöglet – védekezés", BOARD)
        assert meta["kind"] == "board" and meta["name"] == "Szöglet – védekezés"
        assert meta["id"].startswith("szoglet-vedekezes-"), meta["id"]
        rep = st.save("Elemzés 2. félidő", REPLAY)
        assert rep["kind"] == "replay"  # a 'stops' alapján
        lst = st.list()
        assert {b["id"] for b in lst} == {meta["id"], rep["id"]}
        assert [b["id"] for b in st.list(kind="replay")] == [rep["id"]]
        assert st.list(kind="replay")[0]["events_url"] == "/clips/match_events_ab12.json"
        doc = st.get(meta["id"])
        assert doc["data"] == BOARD and doc["name"] == "Szöglet – védekezés"

        # felülírás: ugyanaz az id, a created_at marad
        newer = dict(BOARD, board=dict(BOARD["board"], form1="4-4-2"))
        up = st.save("Átnevezve", newer, board_id=meta["id"])
        assert up["id"] == meta["id"] and up["created_at"] == meta["created_at"]
        assert st.get(meta["id"])["data"]["board"]["form1"] == "4-4-2"
        assert len(st.list()) == 2

        assert st.delete(meta["id"]) is True
        _raises(lambda: st.get(meta["id"]), 404)
        _raises(lambda: st.delete(meta["id"]), 404)
        assert [b["id"] for b in st.list()] == [rep["id"]]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_ids_are_sanitized():
    d, st = _store()
    try:
        for bad in ("../titok", "a/b", "", "x" * 65, "tábla", "a.json", None, 5):
            _raises(lambda b=bad: st.get(b), 400)
        _raises(lambda: st.save("x", BOARD, board_id="../../etc/passwd"), 400)
        _raises(lambda: st.delete("..\\x"), 400)
        # a mappán kívül nem jött létre semmi
        assert os.listdir(d) == ["boards"]
        # a név tisztítása: vezérlőkarakterek, hossz, üres név
        meta = st.save("  sor\x00tö\nrés\t  ", BOARD)
        assert meta["name"] == "sor tö rés"
        assert st.save("", BOARD)["name"] == "Névtelen tábla"
        assert len(st.save("h" * 500, BOARD)["name"]) == boards.MAX_NAME_LEN
        # idegen fájl a mappában nem zavarja a listát
        with open(os.path.join(st.root, "rossz.json"), "w") as f:
            f.write("{nem json")
        with open(os.path.join(st.root, "jegyzet.txt"), "w") as f:
            f.write("x")
        assert len(st.list()) == 3
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_validation_and_size_limit():
    d, st = _store(max_bytes=2000, max_boards=3)
    try:
        _raises(lambda: st.save("x", [1, 2]), 400)
        _raises(lambda: st.save("x", {"board": {"players": "nem lista"}}), 400)
        _raises(lambda: st.save("x", {"stops": [{"t": "később", "items": []}]}), 400)
        _raises(lambda: st.save("x", {"stops": "x"}, kind="replay"), 400)
        _raises(lambda: st.save("x", BOARD, kind="video"), 400)
        big = {"board": {"players": [], "drawings": [{"type": "free", "pts": [[i, i] for i in range(500)]}]}}
        ex = _raises(lambda: st.save("nagy", big), 413)
        assert "túl nagy" in ex.message
        assert st.list() == []  # semmi nem maradt (tmp fájl sem)
        assert os.listdir(st.root) == []
        for i in range(3):
            st.save(f"t{i}", BOARD)
        _raises(lambda: st.save("negyedik", BOARD), 507)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_parse_request_body():
    name, data, bid, kind = parse_request_body(json.dumps({"name": "A", "data": BOARD}).encode("utf-8"))
    assert name == "A" and data == BOARD and bid is None and kind is None
    _, _, bid, kind = parse_request_body(b'{"name":"A","data":{"stops":[]},"id":"abc-1","kind":"replay"}')
    assert bid == "abc-1" and kind == "replay"
    _raises(lambda: parse_request_body(b""), 400)
    _raises(lambda: parse_request_body(b"{nem json"), 400)
    _raises(lambda: parse_request_body(b"[1]"), 400)
    _raises(lambda: parse_request_body(b'{"name":"A"}'), 400)
    _raises(lambda: parse_request_body(b'{"data":{},"id":5}'), 400)
    _raises(lambda: parse_request_body(b"x" * 10000, max_bytes=1000), 413)


def test_saved_file_shape():
    d, st = _store()
    try:
        meta = st.save("Forma", REPLAY)
        with open(os.path.join(st.root, meta["id"] + ".json"), encoding="utf-8") as f:
            doc = json.load(f)
        assert set(doc) == {"id", "name", "kind", "created_at", "updated_at", "data"}
        assert doc["data"]["stops"][0]["t"] == 3712.5
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK    {name}")
        except Exception as ex:
            failed += 1
            print(f"HIBA  {name}: {ex!r}")
    print(f"{len(tests) - failed}/{len(tests)} teszt sikeres")
    sys.exit(1 if failed else 0)
