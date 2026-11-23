"""
Jelenet-címkéző csomag: az elemzés eseményeiből rövid videórészletek + egy önálló weboldal, amelyen az
edző bármelyik gépen (laptopon is, telepítés és internet nélkül, csak böngészővel) megnézi a jeleneteket,
és jelöli: jó-e a program tippje, ha nem, mi történt valójában. A végén a jelölések egy fájlba mentődnek,
amit a program visszaolvas (a mérésekhez és a későbbi tanításhoz).

1) Csomag készítése (azon a gépen, ahol a meccsvideó és az elemzés van):
    python tools\\cimkezo_csomag.py keszit --esemenyek clips\\match_events_ab12.json --ki cimkezo_meccs1
    (további: --veletlen 20  -> 20 véletlen jelenet is, amit a program nem jelölt meg;
              --szelesseg 960, --elotte 5, --utana 4, --video D:\\Veo\\meccs.mp4)

2) A "cimkezo_meccs1" mappát másold a laptopra (pendrive / Google Drive), és nyisd meg benne az
   index.html-t (Chrome / Edge). A jelölések menet közben is mentődnek a böngészőben; a végén
   "Mentés fájlba" -> cimkek_<meccs>.json.

3) A címkék visszaolvasása (az elemző gépen):
    python tools\\cimkezo_csomag.py beolvas --cimkek cimkek_match_events_ab12.json
    -> clips\\feedback_match_events_ab12.json (ugyanaz, amit a program ✓ / ✗ gombjai írnak)
"""
import argparse
import html
import json
import os
import random
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

# a címkézhető esemény-típusok (kód, magyar név, kalap) – a "semmi" = nem történt semmi lényeges
TYPES = [
    ("pass", "Passz", "Passz"),
    ("pass_failed", "Rossz passz / labdavesztés", "Passz"),
    ("cross", "Beadás", "Beadás"),
    ("shot", "Lövés", "Befejezés"),
    ("goal", "Gól", "Befejezés"),
    ("corner", "Szöglet", "Rögzített helyzet"),
    ("free_kick", "Szabadrúgás", "Rögzített helyzet"),
    ("throw_in", "Bedobás", "Rögzített helyzet"),
    ("gk_goal_kick", "Kirúgás (alapvonalról)", "Rögzített helyzet"),
    ("kickoff", "Középkezdés", "Rögzített helyzet"),
    ("tackle", "Szerelés", "Labdaszerzés"),
    ("interception", "Labdaszerzés / közbelépés", "Labdaszerzés"),
    ("dribble", "Cselezés / labdavezetés", "Egyéb"),
    ("gk_save", "Kapus: védés", "Kapus"),
    ("gk_claim_cross", "Kapus: beadás leszedése", "Kapus"),
    ("gk_long_ball", "Kapus: ívelés / kirúgás kézből", "Kapus"),
    ("gk_short_pass", "Kapus: rövid passz / kidobás", "Kapus"),
    ("gk_sweeper", "Kapus: kifutás", "Kapus"),
    ("gk_conceded", "Kapus: kapott gól", "Kapus"),
    ("egyeb", "Egyéb (írd a megjegyzésbe)", "Egyéb"),
    ("semmi", "Nem történt semmi lényeges", "Egyéb"),
]
LABELS = {t: l for t, l, _ in TYPES}


def _clip_window(ev, pre, post, max_len=15.0):
    t = float(ev.get("time", 0.0))
    s = float(ev["start_sec"]) if ev.get("start_sec") is not None else t - pre
    e = float(ev["end_sec"]) if ev.get("end_sec") is not None else t + post
    s, e = max(0.0, min(s, t - 2.0)), max(e, t + 2.0)
    if e - s > max_len:   # túl hosszú ablak: az esemény köré szűkítve
        s, e = max(0.0, t - max_len * 0.6), t + max_len * 0.4
    return s, e


def _ffmpeg():
    try:
        import fast_cut
        return fast_cut.resolve_ffmpeg(None)
    except Exception:
        return os.environ.get("FFMPEG_PATH", "ffmpeg")


def cut_small(video, start, end, out_path, width=960, ffmpeg=None):
    """Kis felbontású, gyorsan lejátszható részlet (H.264, hang nélkül) – a csomag kicsi maradjon."""
    ffmpeg = ffmpeg or _ffmpeg()
    enc = "libx264"
    args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p"]
    try:
        from video_writer import choose_h264_encoder, encoder_args
        enc = choose_h264_encoder(ffmpeg)
        args = encoder_args(enc, width, int(width * 9 / 16))
    except Exception:
        pass
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", video,
           "-t", f"{end - start:.3f}", "-vf", f"scale={int(width)}:-2", "-an", *args,
           "-movflags", "+faststart", out_path]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError("ffmpeg hiba: " + res.stderr.decode("utf-8", "replace").strip()[-300:])


def random_times(halves, events, n, pre, post, seed=0):
    """n véletlen időpont a félidőkön belül, a meglévő események ablakaitól távol."""
    rng = random.Random(seed)
    spans = [(float(h["start_sec"]) + pre, float(h["end_sec"]) - post) for h in (halves or {}).values()
             if h and float(h["end_sec"]) - float(h["start_sec"]) > pre + post + 1]
    taken = [float(e.get("time", 0.0)) for e in events]
    out = []
    for _ in range(n * 30):
        if len(out) >= n or not spans:
            break
        a, b = rng.choice(spans)
        t = rng.uniform(a, b)
        if all(abs(t - x) > pre + post for x in taken + out):
            out.append(t)
    return sorted(out)


def doubtful_events(data, events_path, n, log=print):
    """
    "Kétes" jelenetek: amit a detektor engedékenyebb küszöbökkel jelezne, a normál futás viszont nem
    (tools/hangolas.py near_misses) – a detektor-naplóból, vagy ha nincs, a mentett pályákból.
    """
    if n <= 0:
        return []
    try:
        import events as events_mod
        import app_paths
        import hangolas
        segs = hangolas.load_segments(data, os.path.dirname(os.path.abspath(events_path)))
        if not segs:
            return []
        base = events_mod.load_tuning(app_paths.event_tuning_path())
        avoid = [float(e.get("time", 0.0)) for e in data.get("events") or []]
        out = hangolas.near_misses(segs, base, n=n, avoid_times=avoid)
        log(f"Kétes jelenetek: {len(out)} ({'detektor-napló' if segs[0].get('source') == 'napló' else 'pályák'} alapján)")
        return out
    except Exception as ex:   # a csomag kétes jelenetek nélkül is elkészül
        log(f"(kétes jelenetek nélkül: {ex})")
        return []


def build(events_path, out_dir, video=None, n_random=10, pre=5.0, post=4.0, width=960, log=print, n_doubt=0,
          progress=None):
    with open(events_path, encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("events") or []
    video = video or (data.get("source") or {}).get("video_path")
    if not video or not os.path.exists(video):
        raise SystemExit(f"A meccsvideó nem található: {video!r} – add meg: --video <útvonal>")
    halves = data.get("halves") or {}
    os.makedirs(os.path.join(out_dir, "klipek"), exist_ok=True)
    meccs = os.path.splitext(os.path.basename(events_path))[0]
    items = []
    todo = [("esemeny", e) for e in events if e.get("type") != "gk_positioning_error"]
    doubt = doubtful_events(data, events_path, n_doubt, log)
    todo += [("ketes", e) for e in doubt]
    todo += [("minta", {"type": "?", "time": t}) for t in random_times(halves, events + doubt, n_random, pre, post)]
    todo.sort(key=lambda x: float(x[1].get("time", 0.0)))
    for i, (kind, ev) in enumerate(todo, 1):
        s, e = _clip_window(ev, pre, post)
        name = f"klip_{i:03d}.mp4"
        log(f"  [{i}/{len(todo)}] {LABELS.get(ev.get('type'), ev.get('type'))} @ {ev.get('time', 0):.1f} mp")
        if progress is not None:
            progress(i - 1, len(todo))
        cut_small(video, s, e, os.path.join(out_dir, "klipek", name), width)
        items.append({
            "id": i, "file": f"klipek/{name}", "kind": kind, "start": round(s, 2), "end": round(e, 2),
            "time": round(float(ev.get("time", 0.0)), 2), "type": ev.get("type"),
            "label": LABELS.get(ev.get("type"), ev.get("type")) if kind in ("esemeny", "ketes") else "?",
            "team": ev.get("team"), "desc": ev.get("desc", ""),
            "match_clock": ev.get("match_clock"), "half_label": ev.get("half_label"),
        })
    payload = {
        "meccs": meccs, "events_url": f"/clips/{os.path.basename(events_path)}", "video": video,
        "team_colors": data.get("team_colors"), "tipusok": [{"type": t, "label": l, "kalap": k} for t, l, k in TYPES],
        "klipek": items,
    }
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(PAGE.replace("/*ADAT*/null", json.dumps(payload, ensure_ascii=False))
                .replace("{{CIM}}", html.escape(meccs)))
    log(f"Kész: {out_dir} ({len(items)} jelenet). Másold a mappát a laptopra, és nyisd meg az index.html-t.")
    return payload


def import_labels(labels_path, clips_dir=None, log=print):
    import feedback as fb
    import app_paths
    with open(labels_path, encoding="utf-8") as f:
        lab = json.load(f)
    clips_dir = clips_dir or app_paths.clips_dir()
    url = lab["events_url"]
    n_ok = n_bad = n_new = 0
    for c in lab.get("klipek", []):
        v = c.get("cimke")
        if not v:
            continue
        when = c["start"] + float(v["pillanat"]) if v.get("pillanat") is not None else None
        # a végignézett ablak (a "semmi" címke erős negatív minta a hangolásnak)
        if c.get("start") is not None and c.get("end") is not None:
            lab_t = (v.get("tipus") if v.get("itelet") != "ok" else c.get("type")) if v.get("itelet") else None
            fb.add_reviewed(clips_dir, url, c["start"], c["end"], label=lab_t, source="cimkezo",
                            fontos=v.get("fontos") if "fontos" in v else None)
        fontos = v.get("fontos") if "fontos" in v else None
        if not v.get("itelet"):
            # csak "fontos" jelölés, ítélet nélkül: a végignézett ablakhoz kerül
            continue
        if c["kind"] == "esemeny":
            ev = {"type": c["type"], "time": c["time"], "team": c.get("team")}
            if v["itelet"] == "ok":
                fb.set_verdict(clips_dir, url, ev, "ok", corrected_team=v.get("csapat") if v.get("csapat") != c.get("team") else None,
                               note=v.get("megjegyzes"), fontos=fontos)
                n_ok += 1
            else:
                fb.set_verdict(clips_dir, url, ev, "hamis", corrected_type=v.get("tipus"), corrected_team=v.get("csapat"),
                               note=v.get("megjegyzes"), corrected_time=when, fontos=fontos)
                n_bad += 1
        elif c["kind"] == "ketes" and v["itelet"] == "ok":
            # a program "majdnem" jelezte, és tényleg az volt: a normál futásból kimaradt valódi esemény
            fb.add_missing(clips_dir, url, c["type"], when if when is not None else c["time"],
                           v.get("csapat") or c.get("team"), v.get("megjegyzes") or "kétes jelenet: igen", fontos=fontos)
            n_new += 1
        elif v.get("tipus") and v["tipus"] != "semmi":
            fb.add_missing(clips_dir, url, v["tipus"], when if when is not None else c["time"], v.get("csapat"),
                           v.get("megjegyzes", ""), fontos=fontos)
            n_new += 1
    log(f"Beolvasva: {n_ok} jó, {n_bad} téves/javított, {n_new} új (a program által nem jelölt) esemény -> "
        f"{fb.feedback_path(clips_dir, url)}")
    return n_ok, n_bad, n_new


def main(argv=None):
    ap = argparse.ArgumentParser(description="Jelenet-címkéző csomag (videórészletek + offline weboldal)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keszit", help="csomag készítése egy elemzés eseményeiből")
    k.add_argument("--esemenyek", required=True, help="clips\\match_events_*.json (vagy events_*.json)")
    k.add_argument("--ki", required=True, help="kimeneti mappa")
    k.add_argument("--video", help="a meccsvideó (alap: az elemzésben tárolt útvonal)")
    k.add_argument("--veletlen", type=int, default=10, help="ennyi véletlen, nem jelölt jelenet is (alap 10)")
    k.add_argument("--ketes", type=int, default=15,
                   help="ennyi kétes jelenet: amit a program majdnem jelzett (detektor-napló vagy pályák alapján; alap 15)")
    k.add_argument("--elotte", type=float, default=5.0)
    k.add_argument("--utana", type=float, default=4.0)
    k.add_argument("--szelesseg", type=int, default=960, help="a részletek szélessége képpontban (alap 960)")
    b = sub.add_parser("beolvas", help="a laptopon készült cimkek_*.json beolvasása")
    b.add_argument("--cimkek", required=True)
    b.add_argument("--clips", help="a clips mappa (alap: a program adatmappája)")
    b.add_argument("--hangol", action="store_true",
                   help="a beolvasás után a küszöbök hangolása is az összes eddigi címkéből (tools/hangolas.py), "
                        "és mentés: a következő elemzés már ezekkel fut")
    a = ap.parse_args(argv)
    if a.cmd == "keszit":
        build(a.esemenyek, a.ki, a.video, a.veletlen, a.elotte, a.utana, a.szelesseg, n_doubt=a.ketes)
    else:
        import_labels(a.cimkek, a.clips)
        if a.hangol:
            return tune_all(a.clips)
    return 0


def tune_all(clips_dir=None):
    """Hangolás az összes olyan meccsből, amelyhez van visszajelzés (feedback_*.json) a clips mappában."""
    import hangolas
    args = ["--clips", clips_dir] if clips_dir else []
    print("\nHangolás az eddigi címkékből…")
    return hangolas.main(args + ["--ment"])


PAGE = r"""<!doctype html>
<html lang="hu"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jelenet-címkéző – {{CIM}}</title>
<style>
:root { --bg:#111; --panel:#1c1c1c; --fg:#eee; --muted:#9a9a9a; --ok:#2e7d32; --bad:#c62828; --acc:#ff7a2f; }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--fg); font:14px system-ui, Segoe UI, sans-serif; }
header { display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:space-between; padding:10px 14px; background:#181818; }
h1 { font-size:18px; margin:0; } .muted { color:var(--muted); } button, select, input, textarea { font:inherit; }
button { background:#333; color:var(--fg); border:1px solid #444; border-radius:7px; padding:7px 12px; cursor:pointer; }
button.ok { background:var(--ok); border-color:var(--ok); } button.bad { background:var(--bad); border-color:var(--bad); }
button.acc { background:var(--acc); border-color:var(--acc); color:#1a0d04; font-weight:600; }
main { display:grid; grid-template-columns: 300px minmax(0,1fr); gap:12px; padding:12px; }
@media (max-width:900px) { main { grid-template-columns: 1fr; } }
#lista { background:var(--panel); border-radius:10px; padding:6px; max-height:calc(100vh - 90px); overflow:auto; }
.sor { padding:7px 9px; border-radius:7px; cursor:pointer; display:flex; justify-content:space-between; gap:8px; }
.sor:hover { background:#262626; } .sor.akt { outline:2px solid var(--acc); }
.sor .t { font-variant-numeric:tabular-nums; color:var(--muted); } .jel { font-size:15px; }
#nezo { background:var(--panel); border-radius:10px; padding:12px; display:flex; flex-direction:column; gap:10px; }
video { width:100%; max-height:60vh; background:#000; border-radius:8px; }
.tipp { font-size:17px; } .tipp b { color:var(--acc); }
.gombok { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
#javit { display:none; flex-direction:column; gap:8px; background:#222; padding:10px; border-radius:8px; }
select, input, textarea { background:#111; color:var(--fg); border:1px solid #444; border-radius:6px; padding:6px 8px; }
textarea { width:100%; min-height:54px; } .kbd { font-size:12px; color:var(--muted); }
</style></head><body>
<header>
  <div><h1>Jelenet-címkéző</h1><div class="muted" id="alcim"></div></div>
  <div class="gombok">
    <span id="halad" class="muted"></span>
    <button class="acc" id="ment">💾 Mentés fájlba</button>
    <label class="muted">Betöltés: <input type="file" id="betolt" accept=".json"></label>
  </div>
</header>
<main>
  <div id="lista"></div>
  <div id="nezo">
    <video id="vid" controls playsinline loop></video>
    <div class="tipp" id="tipp"></div>
    <div class="muted" id="leiras"></div>
    <div class="gombok">
      <button class="ok" id="jo">✓ Jó (J)</button>
      <button class="bad" id="rossz">✗ Nem jó / mi volt? (N)</button>
      <button id="elozo">‹ Előző</button><button id="kov">Következő ›</button>
      <button id="lassu">0,5×</button>
      <button id="fontos" title="Ezt a jelenetet látni akarom a válogatásban">☆ Fontos (F)</button>
    </div>
    <div id="javit">
      <div class="gombok"><label>Mi történt valójában? <select id="tipus"></select></label>
        <label>Csapat: <select id="csapat"><option value="">nem tudom</option><option value="team1">1. csapat</option><option value="team2">2. csapat</option></select></label></div>
      <div class="gombok"><button id="most">⏱ Ebben a pillanatban történt</button><span class="muted" id="pill"></span></div>
      <textarea id="megj" placeholder="Megjegyzés (nem kötelező) – pl. „hosszú passz a jobb szélre”"></textarea>
      <div class="gombok"><button class="acc" id="rogzit">Rögzítés (Enter)</button></div>
    </div>
    <div class="kbd">Billentyűk: J = jó, N = nem jó, F = fontos, ← / → = előző / következő, szóköz = lejátszás. A jelölések a böngészőben is
      mentődnek; a végén a „Mentés fájlba” gombbal mentsd le, és a fájlt vidd vissza az elemző gépre.</div>
  </div>
</main>
<script>
const ADAT = /*ADAT*/null;
const KULCS = "cimkezo:" + ADAT.meccs;
let cimkek = {}; try { cimkek = JSON.parse(localStorage.getItem(KULCS) || "{}"); } catch (e) {}
let fontos = {}; try { fontos = JSON.parse(localStorage.getItem(KULCS + ":fontos") || "{}"); } catch (e) {}
let akt = 0, pill = null;
const $ = (id) => document.getElementById(id);
const csapatNev = (t) => t === "team1" ? "1. csapat" : t === "team2" ? "2. csapat" : "";
const fmt = (s) => { s = Math.max(0, Math.round(s)); return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); };
$("alcim").textContent = ADAT.meccs + " • " + ADAT.klipek.length + " jelenet";
for (const t of ADAT.tipusok) { const o = document.createElement("option"); o.value = t.type; o.textContent = t.kalap + " – " + t.label; $("tipus").appendChild(o); }
function ment() { try { localStorage.setItem(KULCS, JSON.stringify(cimkek)); localStorage.setItem(KULCS + ":fontos", JSON.stringify(fontos)); } catch (e) {} lista(); }
function fontosGomb() { const f = !!fontos[ADAT.klipek[akt].id]; $("fontos").textContent = f ? "⭐ Fontos (F)" : "☆ Fontos (F)"; $("fontos").className = f ? "acc" : ""; }
function jel(k) { const c = cimkek[k.id]; if (!c) return "·"; return c.itelet === "ok" ? "✅" : (c.tipus === "semmi" ? "⛔" : "✏️"); }
function lista() {
  const L = $("lista"); L.innerHTML = "";
  ADAT.klipek.forEach((k, i) => {
    const d = document.createElement("div"); d.className = "sor" + (i === akt ? " akt" : "");
    const nev = k.kind === "minta" ? "❓ (véletlen jelenet)" : k.kind === "ketes" ? "🤔 " + k.label + "?" : k.label;
    d.innerHTML = `<span><span class="t">${fmt(k.time)}</span> ${nev}${k.team ? " – " + csapatNev(k.team) : ""}</span><span class="jel">${fontos[k.id] ? "⭐" : ""}${jel(k)}</span>`;
    d.onclick = () => mutat(i); L.appendChild(d);
  });
  const n = Object.keys(cimkek).length; $("halad").textContent = `${n} / ${ADAT.klipek.length} megjelölve`;
}
function mutat(i) {
  akt = Math.max(0, Math.min(ADAT.klipek.length - 1, i)); const k = ADAT.klipek[akt]; pill = null;
  $("vid").src = k.file; $("vid").playbackRate = 1; $("vid").play().catch(() => {});
  $("tipp").innerHTML = k.kind === "minta" ? "Véletlen jelenet – a program nem jelölt itt semmit. <b>Mi történik?</b>"
    : k.kind === "ketes" ? `Kétes jelenet – a program <b>majdnem</b> ezt jelölte: <b>${k.label}</b>${k.team ? " – " + csapatNev(k.team) : ""}. Az volt? <span class="muted">(${fmt(k.time)} a videóban)</span>`
    : `A program tippje: <b>${k.label}</b>${k.team ? " – " + csapatNev(k.team) : ""} <span class="muted">(${fmt(k.time)} a videóban${k.match_clock ? ", " + k.match_clock + " meccsidő" : ""})</span>`;
  $("leiras").textContent = k.desc || "";
  const c = cimkek[k.id];
  $("javit").style.display = k.kind === "minta" || (c && c.itelet === "hamis") ? "flex" : "none";
  $("tipus").value = (c && c.tipus) || (k.kind === "minta" ? "semmi" : k.type);
  $("csapat").value = (c && c.csapat) || k.team || ""; $("megj").value = (c && c.megjegyzes) || "";
  $("pill").textContent = c && c.pillanat != null ? "pillanat: " + c.pillanat.toFixed(1) + " mp a részletben" : "";
  $("jo").disabled = k.kind === "minta";
  fontosGomb(); lista();
}
$("jo").onclick = () => { const k = ADAT.klipek[akt]; if (k.kind === "minta") return; cimkek[k.id] = { itelet: "ok" }; ment(); mutat(akt + 1); };
$("rossz").onclick = () => { $("javit").style.display = "flex"; if (ADAT.klipek[akt].kind !== "minta" && $("tipus").value === ADAT.klipek[akt].type) $("tipus").value = "semmi"; $("tipus").focus(); };
$("fontos").onclick = () => { const k = ADAT.klipek[akt]; if (fontos[k.id]) delete fontos[k.id]; else fontos[k.id] = true; ment(); fontosGomb(); };
$("most").onclick = () => { pill = $("vid").currentTime; $("pill").textContent = "pillanat: " + pill.toFixed(1) + " mp a részletben"; };
$("rogzit").onclick = () => {
  const k = ADAT.klipek[akt]; const prev = cimkek[k.id] || {};
  cimkek[k.id] = { itelet: k.kind === "minta" ? "minta" : "hamis", tipus: $("tipus").value, csapat: $("csapat").value || null,
                   megjegyzes: $("megj").value.trim() || null, pillanat: pill != null ? pill : (prev.pillanat ?? null) };
  ment(); mutat(akt + 1);
};
$("elozo").onclick = () => mutat(akt - 1); $("kov").onclick = () => mutat(akt + 1);
$("lassu").onclick = () => { const v = $("vid"); v.playbackRate = v.playbackRate === 1 ? 0.5 : 1; $("lassu").textContent = v.playbackRate === 1 ? "0,5×" : "1×"; };
document.addEventListener("keydown", (e) => {
  if (["TEXTAREA", "INPUT", "SELECT"].includes(document.activeElement.tagName)) { if (e.key === "Enter" && document.activeElement.tagName !== "TEXTAREA") $("rogzit").click(); return; }
  if (e.key === "j" || e.key === "J") $("jo").click(); else if (e.key === "n" || e.key === "N") $("rossz").click();
  else if (e.key === "f" || e.key === "F") $("fontos").click();
  else if (e.key === "ArrowRight") mutat(akt + 1); else if (e.key === "ArrowLeft") mutat(akt - 1);
  else if (e.key === "Enter") $("rogzit").click();
  else if (e.key === " ") { e.preventDefault(); const v = $("vid"); v.paused ? v.play() : v.pause(); }
});
$("ment").onclick = () => {
  const cimkeKi = (k) => cimkek[k.id] ? { ...cimkek[k.id], fontos: !!fontos[k.id] } : (fontos[k.id] ? { fontos: true } : null);
  const ki = { meccs: ADAT.meccs, events_url: ADAT.events_url, klipek: ADAT.klipek.map((k) => ({ ...k, cimke: cimkeKi(k) })) };
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(ki, null, 1)], { type: "application/json" }));
  a.download = "cimkek_" + ADAT.meccs + ".json"; a.click();
};
$("betolt").onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  try {
    const d = JSON.parse(await f.text());
    for (const k of d.klipek || []) {
      if (!k.cimke) continue;
      const { fontos: fo, ...rest } = k.cimke;
      if (fo) fontos[k.id] = true;
      if (rest.itelet) cimkek[k.id] = rest;
    }
    ment(); mutat(akt);
  }
  catch (err) { alert("A fájl nem olvasható: " + err.message); }
};
mutat(0);
</script></body></html>
"""

if __name__ == "__main__":
    sys.exit(main())
