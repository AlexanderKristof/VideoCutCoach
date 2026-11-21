"""
Áttekintő export ("ellenőrző lapok") az elemzett videóhoz.

Cél: az elemzés eredményét kevés képből is át lehessen nézni (akár
telefonról, akár az AI által), anélkül, hogy a teljes videót kellene nézni.

- Esemény-lap: minden felismert eseménynél (lövés, védés, gól, ...)
  egyetlen JPG készül, rajta az esemény előtti és utáni pillanatok
  rácsban (pl. 3x3 kép), időbélyeggel, a kirajzolt dobozokkal, és
  egy 2D minitérképpel a játékosok pozíciójáról.
- Áttekintő lap: adott időközönként (pl. percenként) egy-egy kép,
  lapanként több pillanat – így a teljes meccs pár képen átfutható.
- index.json + index.html: az összes lap listája az eseményadatokkal.
"""
import os
import json
import html
from collections import deque

import cv2
import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

TEAM_COLORS = {
    1: (0, 0, 255),      # piros (BGR)
    2: (255, 0, 0),      # kék
}
KEEPER_COLOR = (0, 165, 255)  # narancs
BALL_COLOR = (255, 255, 255)


def draw_minimap(players, ball=None, width=420):
    """
    2D felülnézeti pálya tiszta színekkel.
    players: [{'pitch_coords': (x, y), 'team_id': 1/2, 'is_keeper': bool, 'track_id': int}, ...]
    ball: (x, y) méterben vagy None
    """
    scale = width / PITCH_LENGTH
    height = int(PITCH_WIDTH * scale)
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = (60, 130, 60)

    def pt(x, y):
        return int(x * scale), int(y * scale)

    line = (235, 235, 235)
    cv2.rectangle(img, pt(0, 0), pt(PITCH_LENGTH, PITCH_WIDTH), line, 1)
    cv2.line(img, pt(PITCH_LENGTH / 2, 0), pt(PITCH_LENGTH / 2, PITCH_WIDTH), line, 1)
    cv2.circle(img, pt(PITCH_LENGTH / 2, PITCH_WIDTH / 2), int(9.15 * scale), line, 1)
    # 16-osok és ötösök
    for x0, sign in ((0.0, 1), (PITCH_LENGTH, -1)):
        cv2.rectangle(img, pt(x0, 13.84), pt(x0 + sign * 16.5, 54.16), line, 1)
        cv2.rectangle(img, pt(x0, 24.84), pt(x0 + sign * 5.5, 43.16), line, 1)
        cv2.line(img, pt(x0, 30.34), pt(x0, 37.66), (255, 255, 255), 3)

    for p in players or []:
        coords = p.get('pitch_coords')
        if not coords:
            continue
        color = KEEPER_COLOR if p.get('is_keeper') else TEAM_COLORS.get(p.get('team_id'), (200, 200, 200))
        c = pt(*coords)
        cv2.circle(img, c, 5, color, -1)
        cv2.circle(img, c, 5, (0, 0, 0), 1)

    if ball:
        cv2.circle(img, pt(*ball), 3, BALL_COLOR, -1)
        cv2.circle(img, pt(*ball), 3, (0, 0, 0), 1)
    return img


def _fit(img, w, h):
    """Kép beillesztése w x h cellába arányosan, fekete kerettel."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    resized = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_AREA)
    cell = np.zeros((h, w, 3), np.uint8)
    y0 = (h - resized.shape[0]) // 2
    x0 = (w - resized.shape[1]) // 2
    cell[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return cell


def _label(img, text, org=(6, 20), scale=0.55):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _ascii(text):
    """Az OpenCV betűkészlete nem ismeri az ékezeteket – a képre ékezet nélkül írunk."""
    table = str.maketrans("áéíóöőúüűÁÉÍÓÖŐÚÜŰ–—", "aeiooouuuAEIOOOUUU--")
    return text.translate(table).encode("ascii", "replace").decode("ascii")


def fmt_time(sec):
    sec = max(0.0, float(sec))
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:04.1f}"


def make_contact_sheet(tiles, title, cols=3, tile_w=520, tile_h=293):
    """
    tiles: [(kép, felirat), ...] → egyetlen rácsos kép címsorral.
    """
    rows = max(1, (len(tiles) + cols - 1) // cols)
    header = 40
    sheet = np.zeros((header + rows * tile_h, cols * tile_w, 3), np.uint8)
    sheet[:header] = (40, 40, 40)
    _label(sheet, _ascii(title), (10, 27), 0.7)
    for i, (img, caption) in enumerate(tiles):
        r, c = divmod(i, cols)
        cell = _fit(img, tile_w - 4, tile_h - 4)
        if caption:
            _label(cell, _ascii(caption))
        y = header + r * tile_h + 2
        x = c * tile_w + 2
        sheet[y:y + tile_h - 4, x:x + tile_w - 4] = cell
    return sheet


class ReviewExporter:
    """
    Használat a feldolgozó ciklusban:
        rx = ReviewExporter(out_dir, fps)
        minden képkockánál: rx.add_frame(idx, ido_mp, annotalt_frame, players, ball)
        eseménynél:         rx.add_event({"type": ..., "time": ..., "desc": ...})
        a végén:            rx.close()  → index.json, index.html
    """

    def __init__(self, out_dir, fps, pre_sec=3.0, post_sec=3.0, tiles_per_event=9,
                 overview_every_sec=60.0, overview_per_sheet=12, preview_width=960,
                 jpeg_quality=82):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.fps = fps or 25.0
        self.pre_sec = pre_sec
        self.post_sec = post_sec
        self.tiles_per_event = max(2, tiles_per_event)
        self.overview_every_sec = overview_every_sec
        self.overview_per_sheet = overview_per_sheet
        self.preview_width = preview_width
        self.jpeg_quality = jpeg_quality

        # Ennyi időnként teszünk el kis képet a pufferbe (az esemény előtti részhez)
        window = pre_sec + post_sec
        self.sample_step = max(1, int(round(self.fps * window / (self.tiles_per_event - 1))))
        self.buffer = deque(maxlen=self.tiles_per_event + 2)
        self.pending = []           # befejezésre váró esemény-lapok
        self.overview_tiles = []
        self.next_overview_t = None
        self.sheets = []            # index bejegyzések
        self._overview_count = 0
        self._last = None           # utolsó képkocka (az esemény pontos pillanatához)

    # ---- belső segédek -------------------------------------------------

    def _preview(self, frame, players, ball):
        h, w = frame.shape[:2]
        s = self.preview_width / float(w)
        small = cv2.resize(frame, (self.preview_width, int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame.copy()
        if players is not None:
            mm = draw_minimap(players, ball, width=int(small.shape[1] * 0.28))
            mh, mw = mm.shape[:2]
            y0 = small.shape[0] - mh - 6
            x0 = small.shape[1] - mw - 6
            if y0 > 0 and x0 > 0:
                small[y0:y0 + mh, x0:x0 + mw] = mm
                cv2.rectangle(small, (x0 - 1, y0 - 1), (x0 + mw, y0 + mh), (255, 255, 255), 1)
        return small

    def _write(self, name, img):
        path = os.path.join(self.out_dir, name)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return path

    def _finish_event(self, ev):
        closest = min(ev['tiles'], key=lambda ti: abs(ti[0] - ev['time']))[0]
        tiles = [(img, f"{fmt_time(t)}{'  << ESEMENY' if t == closest else ''}") for t, img in ev['tiles']]
        title = f"{ev['type']} @ {fmt_time(ev['time'])} - {ev.get('desc', '')}"
        sheet = make_contact_sheet(tiles, title)
        idx = len([s for s in self.sheets if s['kind'] == 'event']) + 1
        name = f"esemeny_{idx:03d}_{ev['type']}_{int(ev['time'])}s.jpg"
        self._write(name, sheet)
        self.sheets.append({
            "kind": "event", "file": name, "type": ev['type'], "time": round(ev['time'], 2),
            "desc": ev.get('desc', ''), "team": ev.get('team'),
        })

    def _flush_overview(self):
        if not self.overview_tiles:
            return
        self._overview_count += 1
        t0 = self.overview_tiles[0][0]
        t1 = self.overview_tiles[-1][0]
        title = f"Attekintes {self._overview_count}: {fmt_time(t0)} - {fmt_time(t1)}"
        sheet = make_contact_sheet([(img, fmt_time(t)) for t, img in self.overview_tiles], title, cols=4,
                                   tile_w=480, tile_h=270)
        name = f"attekintes_{self._overview_count:03d}.jpg"
        self._write(name, sheet)
        self.sheets.append({"kind": "overview", "file": name, "time": round(t0, 2), "time_end": round(t1, 2)})
        self.overview_tiles = []

    # ---- nyilvános API -------------------------------------------------

    def add_frame(self, frame_idx, time_sec, frame, players=None, ball=None):
        self._last = (time_sec, frame, players, ball)
        need_buffer = frame_idx % self.sample_step == 0
        need_overview = self.overview_every_sec > 0 and (self.next_overview_t is None or time_sec >= self.next_overview_t)
        need_pending = any(time_sec >= ev['next_t'] for ev in self.pending)
        if not (need_buffer or need_overview or need_pending):
            return

        preview = self._preview(frame, players, ball)

        if need_buffer:
            self.buffer.append((time_sec, preview))

        if need_overview:
            self.overview_tiles.append((time_sec, preview))
            self.next_overview_t = time_sec + self.overview_every_sec
            if len(self.overview_tiles) >= self.overview_per_sheet:
                self._flush_overview()

        still_pending = []
        for ev in self.pending:
            if time_sec >= ev['next_t']:
                ev['tiles'].append((time_sec, preview))
                ev['next_t'] = time_sec + self.sample_step / self.fps
            if time_sec >= ev['time'] + self.post_sec or len(ev['tiles']) >= self.tiles_per_event:
                self._finish_event(ev)
            else:
                still_pending.append(ev)
        self.pending = still_pending

    def add_event(self, event):
        """event: {"type", "time", "desc", ...} – az esemény előtti képeket a pufferből veszi."""
        t = float(event.get("time", 0.0))
        pre = [(bt, img) for bt, img in self.buffer if t - self.pre_sec - 1e-6 <= bt <= t]
        pre = pre[-(self.tiles_per_event // 2):]
        # Az esemény pontos pillanata (az utolsó feldolgozott képkocka), ha még nincs benne
        if self._last is not None and abs(self._last[0] - t) <= 1.0 and (not pre or pre[-1][0] < self._last[0]):
            lt, lframe, lplayers, lball = self._last
            pre.append((lt, self._preview(lframe, lplayers, lball)))
        ev = dict(event)
        ev['time'] = t
        ev['tiles'] = list(pre)
        ev['next_t'] = (pre[-1][0] if pre else t) + self.sample_step / self.fps
        self.pending.append(ev)

    @staticmethod
    def _quality_html(q):
        """A videóminőség-összesítő (video_quality.VideoQualityMonitor.summary()) sávja az index.html tetején."""
        if not q or not q.get("grade"):
            return ""
        colors = {"A": "#1f6f3a", "B": "#6f6a1f", "C": "#8a4d12", "D": "#7a1f1f"}
        issues = [f"{html.escape(k)}: {v:.0f}%" for k, v in (q.get("issue_time_pct") or {}).items() if v > 0]
        segs = []
        for sg in (q.get("worst_segments") or [])[:5]:
            segs.append(f"<li>{fmt_time(sg['start_sec'])}–{fmt_time(sg['end_sec'])}: {html.escape(sg['label'])} "
                        f"(megbízhatóság {sg['min_reliability']:.2f})</li>")
        return (f"<div class='q' style='background:{colors.get(q['grade'], '#333')}'>"
                f"<b>Videóminőség: {html.escape(q['grade'])} – {html.escape(q.get('label', ''))}</b> · "
                f"átlagos megbízhatóság {q.get('mean_reliability', 0):.2f}"
                + (f"<br>A klip idejéből képhibás: {', '.join(issues)}" if issues else "<br>Jelentős képhiba nem volt.")
                + (f"<ul>{''.join(segs)}</ul>" if segs else "")
                + "<small>A gyenge minőségű szakaszokon az események megbízhatósága kisebb "
                  "(lásd az események 'reliability' mezőjét).</small></div>")

    def close(self, video_quality=None):
        """Lezárás: index.json / index.html. video_quality: a minőség-összesítő (sáv az index tetején)."""
        for ev in self.pending:
            if ev['tiles']:
                self._finish_event(ev)
        self.pending = []
        self._flush_overview()
        self.sheets.sort(key=lambda s: (s['kind'] != 'overview', s['time']))

        with open(os.path.join(self.out_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"sheets": self.sheets, "video_quality": video_quality}, f, ensure_ascii=False, indent=2)

        rows = []
        for s in self.sheets:
            label = "Áttekintés" if s['kind'] == 'overview' else f"{s['type']} – {html.escape(s.get('desc') or '')}"
            rows.append(f"<section><h3>{fmt_time(s['time'])} · {label}</h3>"
                        f"<img src='{html.escape(s['file'])}' loading='lazy'></section>")
        page = ("<!doctype html><html lang='hu'><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                "<title>Elemzés áttekintő</title><style>body{font-family:sans-serif;background:#111;color:#eee;"
                "margin:0;padding:12px}img{width:100%;height:auto;border:1px solid #444}"
                "h3{font-size:15px;margin:18px 0 6px}.q{padding:10px 14px;border-radius:6px;margin:8px 0 16px;"
                "font-size:14px;line-height:1.5}.q ul{margin:6px 0 4px 18px;padding:0}</style></head><body>"
                "<h1>Elemzés áttekintő</h1>" + self._quality_html(video_quality)
                + "".join(rows) + "</body></html>")
        with open(os.path.join(self.out_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(page)
        return self.sheets
