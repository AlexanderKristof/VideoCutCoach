"""
Ellenőrzés VALÓDI Veo-mintán (nincs a repóban – a felhasználó adja): a VALOS_MINTA_DIR környezeti
változó mappájában kep_*.jpg (1920x1080, percenként egy kép) és klip_*.mp4 (15 mp-es 1080p klipek).
Ha a mappa nincs meg, a teszt kihagyva (0-s kilépési kóddal) ér véget.

    VALOS_MINTA_DIR=/út/valos/minta python3 tests/test_real_samples.py

Mit néz:
  * videóminőség a klipeken: borús / késő délutáni nappali felvétel -> A vagy B osztályzat, és NEM
    "beégett / napfény" + "sötét / zajos" (a fehér égbolt és a sötétebb természetes fű nem képhiba);
  * a képeken a beégés / sötétség címke ritka (a JPEG-minták kockásodása nem számít: az a mintakép
    tömörítéséből jön, nem a videóból).
"""
import os
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

D = os.environ.get("VALOS_MINTA_DIR", "")
if not D or not os.path.isdir(D):
    print("KIHAGYVA: nincs valódi minta (VALOS_MINTA_DIR)")
    sys.exit(0)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import video_quality as vq  # noqa: E402
from pitch_filters import GrassModel  # noqa: E402


def test_clips_grade():
    clips = sorted(glob.glob(os.path.join(D, "klip_*.mp4")))
    assert clips, "nincs klip_*.mp4"
    for path in clips:
        gm = GrassModel.learn_from_video(path)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        q = vq.VideoQualityMonitor(fps, grass_model=gm)
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if q.due(i):
                q.update(f, i / fps)
            i += 1
        s = q.summary()
        print(f"  {os.path.basename(path)}: {s['summary_text']}")
        assert s["grade"] in ("A", "B"), s["summary_text"]
        pct = s["issue_time_pct"]
        assert pct["beégett / napfény"] < 20 and pct["sötét / zajos"] < 20, pct


def test_frames_no_false_glare_dark():
    files = sorted(glob.glob(os.path.join(D, "kep_*.jpg")))
    if not files:
        return
    frames = [cv2.imread(f) for f in files]
    gm = GrassModel()
    for f in frames[:3]:
        gm.observe(f)
    gm.flush()
    q = vq.VideoQualityMonitor(2, grass_model=gm)
    for i, f in enumerate(frames):
        q.update(f, i * 0.5)
    glare = np.mean(["glare" in s["flags"] for s in q.samples])
    dark = np.mean(["dark_noise" in s["flags"] for s in q.samples])
    print(f"  {len(frames)} kép: beégett {glare * 100:.0f}%, sötét/zajos {dark * 100:.0f}%")
    assert glare < 0.1 and dark < 0.1, (glare, dark)


if __name__ == "__main__":
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK   {name}")
        except Exception as ex:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"HIBA {name}: {ex!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} teszt sikeres")
    sys.exit(1 if failed else 0)
