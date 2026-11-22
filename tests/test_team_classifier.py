"""
Automatikus csapatba sorolás tesztjei. Futtatás: python3 tests/test_team_classifier.py
(pytest nem kell; torch/ultralytics nélkül fut, a main.py-t nem importálja)

Szintetikus játékos-kivágások: füves háttér, fej/kar (bőr), mez, nadrág, lábszár;
véletlen méret (20–120 px), árnyék, csillanás, mozgási elmosódás, zaj,
megvilágítás-ingadozás, részleges takarás.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

import team_classifier as tc

GRASS_LO = np.array([30, 40, 40])
GRASS_HI = np.array([85, 255, 255])
SKIN = (120, 160, 210)  # BGR

# BGR mezszínek
KITS = {
    "red": (40, 40, 200), "white": (235, 235, 235),
    "blue": (170, 60, 20), "sky": (230, 190, 120),
    "black": (25, 25, 25), "navy": (50, 28, 24),   # nehéz pár: nagyon sötétkék
    "ref": (30, 230, 240),            # fluoreszkáló sárga (fekete nadrággal)
    "gk_green": (60, 235, 100), "gk_orange": (20, 140, 250),
}


def make_crop(rng, kit, h=None, shorts=(30, 30, 30), occluder=None, corrupt=False):
    """Egy játékos dobozának szintetikus kivágása (BGR, uint8)."""
    h = int(h or rng.integers(20, 121))
    w = max(8, int(h * rng.uniform(0.35, 0.5)))
    H, W = h * 2, w * 2
    img = np.empty((H, W, 3), np.float32)
    grass = np.array([40, 130, 55], np.float32) * rng.uniform(0.8, 1.15)
    img[:] = grass
    img += rng.normal(0, 8, img.shape)
    # csíkos fű
    if rng.random() < 0.5:
        img[: H // 2] *= 1.12
    img = np.clip(img, 0, 255).astype(np.uint8)
    cx = W // 2
    kit = np.array(kit, np.float32) * rng.uniform(0.92, 1.08, 3)
    tw = int(W * rng.uniform(0.55, 0.75))
    # lábak, nadrág, mez, karok, fej
    cv2.rectangle(img, (cx - tw // 3, int(H * 0.7)), (cx - tw // 8, H - 1), SKIN, -1)
    cv2.rectangle(img, (cx + tw // 8, int(H * 0.7)), (cx + tw // 3, H - 1), SKIN, -1)
    cv2.rectangle(img, (cx - tw // 2, int(H * 0.55)), (cx + tw // 2, int(H * 0.72)), shorts, -1)
    cv2.rectangle(img, (cx - tw // 2 - W // 10, int(H * 0.2)), (cx - tw // 2, int(H * 0.5)), SKIN, -1)
    cv2.rectangle(img, (cx + tw // 2, int(H * 0.2)), (cx + tw // 2 + W // 10, int(H * 0.5)), SKIN, -1)
    cv2.rectangle(img, (cx - tw // 2, int(H * 0.17)), (cx + tw // 2, int(H * 0.56)), kit.tolist(), -1)
    cv2.circle(img, (cx, int(H * 0.09)), max(2, int(H * 0.08)), SKIN, -1)
    if rng.random() < 0.3:  # mezszám
        cv2.putText(img, str(rng.integers(1, 30)), (cx - tw // 4, int(H * 0.42)),
                    cv2.FONT_HERSHEY_SIMPLEX, H / 250.0, (240, 240, 240) if kit.mean() < 128 else (20, 20, 20),
                    max(1, H // 60))
    img = img.astype(np.float32)
    # árnyék (az egyik oldal sötétebb) / csillanás
    if rng.random() < 0.5:
        grad = np.linspace(rng.uniform(0.5, 0.8), 1.0, W, dtype=np.float32)
        img *= grad[None, :, None] if rng.random() < 0.5 else grad[None, ::-1, None]
    if rng.random() < 0.2:
        cv2.circle(img, (cx + tw // 4, int(H * 0.3)), H // 12, (255, 255, 255), -1)
    # megvilágítás (nap / árnyék a pályán)
    img *= rng.uniform(0.7, 1.2)
    # részleges takarás egy másik játékos által
    if occluder is not None:
        ow = int(W * rng.uniform(0.2, 0.45))
        if rng.random() < 0.5:
            cv2.rectangle(img, (0, int(H * 0.15)), (ow, int(H * 0.6)), occluder, -1)
        else:
            cv2.rectangle(img, (W - ow, int(H * 0.15)), (W - 1, int(H * 0.6)), occluder, -1)
    if corrupt:
        # erős hiba: a felsőtest nagy részét más mez / pálya takarja
        cv2.rectangle(img, (0, int(H * 0.1)), (int(W * rng.uniform(0.6, 0.9)), int(H * 0.65)), occluder or grass.tolist(), -1)
    img = np.clip(img, 0, 255).astype(np.uint8)
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.4:  # mozgási elmosódás
        k = int(rng.integers(3, 7))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        small = cv2.filter2D(small, -1, ker)
    noise = rng.normal(0, rng.uniform(3, 10), small.shape)
    return np.clip(small.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def simulate(kit_a, kit_b, seconds=30.0, hz=8.0, seed=0, with_pitch=True, extras=True,
             occlusion_rate=0.15, swap=False, reference=None):
    """
    22 + bíró meccs-szimuláció. A csapat (kit_a) a bal térfélen, B a jobbon, kapusok a két kapunál.
    Visszatér: (classifier, rekordok listája (valódi szerep, valódi csapat, eredmény, t))
    """
    rng = np.random.default_rng(seed)
    clf = tc.TeamClassifier(grass_lower=GRASS_LO, grass_upper=GRASS_HI, swap_teams=swap, reference_colors=reference)
    people = []
    for i in range(10):
        people.append(dict(tid=i + 1, kit=KITS[kit_a], role="player", team="A", x=rng.uniform(15, 55), y=rng.uniform(5, 63)))
        people.append(dict(tid=i + 11, kit=KITS[kit_b], role="player", team="B", x=rng.uniform(50, 90), y=rng.uniform(5, 63)))
    if extras:
        people.append(dict(tid=21, kit=KITS["gk_green"], role="keeper", team="A", x=5.0, y=34.0))
        people.append(dict(tid=22, kit=KITS["gk_orange"], role="keeper", team="B", x=100.0, y=34.0))
        people.append(dict(tid=23, kit=KITS["ref"], role="referee", team=None, x=52.0, y=34.0, shorts=(20, 20, 20)))
    recs = []
    n = int(seconds * hz)
    for f in range(n):
        t = f / hz
        cam_x = 52.5 + 40 * np.sin(t / 6.0)   # a forgó kamera kb. 50 m-es sávot lát
        for p in people:
            # mozgás (a kapusok a kapu előtt maradnak)
            if p["role"] == "keeper":
                p["x"] = float(np.clip(p["x"] + rng.normal(0, 0.15), 0.5, 104.5) if p["team"] else p["x"])
                p["x"] = min(p["x"], 14.0) if p["team"] == "A" else max(p["x"], 91.0)
            else:
                p["x"] = float(np.clip(p["x"] + rng.normal(0, 0.5), 2, 103))
            p["y"] = float(np.clip(p["y"] + rng.normal(0, 0.5), 1, 67))
            if abs(p["x"] - cam_x) > 28 or rng.random() < 0.1:
                continue
            occ = None
            if rng.random() < occlusion_rate:
                occ = KITS[kit_b] if p["team"] == "A" else KITS[kit_a]
            crop = make_crop(rng, p["kit"], shorts=p.get("shorts", (30, 30, 30)), occluder=occ)
            res = clf.observe(p["tid"], crop, t, pitch_xy=(p["x"], p["y"]) if with_pitch else None)
            recs.append((p["role"], p["team"], res, t, p["tid"]))
    return clf, recs


def _accuracy(recs, after_t, mapping):
    ok = tot = 0
    for role, team, res, t, _ in recs:
        if role != "player" or t < after_t:
            continue
        tot += 1
        ok += res["team_id"] == mapping[team]
    return ok / max(1, tot)


def _extras_ok(recs, after_t):
    """A bíró / kapusok megfigyeléseinek hány %-a NEM került mezőnyjátékosként csapatba."""
    stats = {}
    for role, team, res, t, _ in recs:
        if role == "player" or t < after_t:
            continue
        good = (res["role"] in ("referee", "other")) if role == "referee" else res["is_keeper"]
        s = stats.setdefault(role, [0, 0])
        s[0] += good
        s[1] += 1
    return {k: v[0] / v[1] for k, v in stats.items()}


def test_descriptor_basic_and_speed():
    rng = np.random.default_rng(1)
    crops = [make_crop(rng, KITS["red"]) for _ in range(300)]
    d, lab = tc.jersey_descriptor(crops[0], GRASS_LO, GRASS_HI)
    assert d.shape == (tc.DESC_DIM,) and d.dtype == np.float32
    assert tc.jersey_descriptor(np.zeros((0, 0, 3), np.uint8))[0] is None
    assert tc.jersey_descriptor(np.zeros((5, 3, 3), np.uint8))[0] is None
    # a piros mez a* értéke erősen pozitív
    assert d[1] > 0.5, d
    t0 = time.perf_counter()
    for c in crops:
        tc.jersey_descriptor(c, GRASS_LO, GRASS_HI)
    per = (time.perf_counter() - t0) / len(crops) * 1e3
    print(f"  jersey_descriptor: {per:.3f} ms / detektálás")
    assert per < 1.0, per


def test_green_keeper_not_eaten_by_grass_mask():
    rng = np.random.default_rng(2)
    d, lab = tc.jersey_descriptor(make_crop(rng, KITS["gk_green"], h=80), GRASS_LO, GRASS_HI)
    rgb = tc._lab_to_rgb(lab)
    assert rgb[1] > 150 and rgb[1] > rgb[0] + 60, rgb


def _run_pair(a, b, seed, expect, label):
    clf, recs = simulate(a, b, seed=seed)
    assert clf.fitted
    after = clf.fit_seconds + 1.0
    acc = _accuracy(recs, after, {"A": 1, "B": 2})
    extras = _extras_ok(recs, after)
    print(f"  {label}: csapat-pontosság={acc * 100:.1f}%, {extras}, szétválás={clf.separation:.2f}, "
          f"sorrend={clf.order_by}")
    print(f"    {clf.describe()}")
    if expect:
        assert acc > 0.95, acc
        assert extras["referee"] > 0.8, extras
        assert extras["keeper"] > 0.8, extras
    return clf, acc


def test_easy_red_vs_white():
    clf, _ = _run_pair("red", "white", 3, True, "piros–fehér")
    cols = clf.team_colors()
    assert cols["team1"][0] > 150 and cols["team1"][1] < 100, cols   # piros
    assert min(cols["team2"]) > 150, cols                              # fehér


def test_easy_blue_vs_sky():
    _run_pair("blue", "sky", 4, True, "kék–égszínkék")


def test_hard_black_vs_navy():
    # nehéz pár: csak beszámolunk, de a véletlennél (50%) lényegesen jobbnak kell lennie
    _, acc = _run_pair("black", "navy", 5, False, "fekete–sötétkék (NEHÉZ)")
    # (W_L = 1.5 óta – a valódi Veo-felvétel fehér–kék mezeinek szétválasztásához – a világosság többet
    # számít, így ez a szintetikus, erősen megvilágítás-ingadozó sötét pár kicsit romlott: ~0.81 -> ~0.67)
    assert acc > 0.6, acc


def test_no_pitch_coords_bigger_cluster_first_and_extras_other():
    clf, recs = simulate("red", "white", seed=6, with_pitch=False)
    assert clf.order_by == "size"
    after = clf.fit_seconds + 1.0
    acc = max(_accuracy(recs, after, {"A": 1, "B": 2}), _accuracy(recs, after, {"A": 2, "B": 1}))
    assert acc > 0.95, acc
    # pályakoordináta nélkül a bíró/kapus "egyéb", nem kerül csapatba
    bad = sum(1 for role, _, res, t, _ in recs if role != "player" and t > after and res["team_id"] is not None)
    tot = sum(1 for role, _, res, t, _ in recs if role != "player" and t > after)
    print(f"  pályakoord. nélkül: pontosság={acc * 100:.1f}%, egyéb csapatba sorolva: {bad}/{tot}")
    assert bad / tot < 0.2


def test_swap_teams():
    clf, recs = simulate("red", "white", seed=3, swap=True)
    acc = _accuracy(recs, clf.fit_seconds + 1.0, {"A": 2, "B": 1})
    assert acc > 0.95, acc
    assert clf.team_colors()["team2"][0] > 150


def test_reference_colours_keep_team_identity():
    """2. félidő: a csapat-számozás az 1. félidő mezszíneiből – akár volt térfélcsere, akár nem
    (valós próba: két szakasz ugyanabból a félidőből, a feltételezett csere megfordította a csapatokat)."""
    first, _ = simulate("red", "white", seed=3)
    ref = first.team_colors()                      # team1 = piros (bal oldal), team2 = fehér
    # nincs csere: a piros továbbra is bal oldalon – a swap_teams=True ellenére is team1 marad
    clf, recs = simulate("red", "white", seed=8, swap=True, reference=ref)
    assert clf.order_by == "reference", clf.order_by
    assert _accuracy(recs, clf.fit_seconds + 1.0, {"A": 1, "B": 2}) > 0.95
    # volt csere: a fehér van bal oldalon, de a piros akkor is team1
    clf2, recs2 = simulate("white", "red", seed=9, reference=ref)
    assert _accuracy(recs2, clf2.fit_seconds + 1.0, {"A": 2, "B": 1}) > 0.95
    assert clf2.team_colors()["team1"][0] > 150 and clf2.team_colors()["team1"][1] < 100


def test_late_extras_and_no_extras_in_warmup():
    # a bemelegítés alatt nincs kapus/bíró: a csapatok ne essenek szét, a később megjelenő
    # külön mezek pedig "egyéb" (kiugró) címkét kapjanak
    rng = np.random.default_rng(7)
    clf = tc.TeamClassifier(grass_lower=GRASS_LO, grass_upper=GRASS_HI)
    for f in range(150):
        t = f / 10.0
        for i in range(10):
            clf.observe(i + 1, make_crop(rng, KITS["blue"]), t, pitch_xy=(30 + i, 30))
            clf.observe(i + 11, make_crop(rng, KITS["white"]), t, pitch_xy=(70 + i, 30))
    assert clf.fitted
    assert sum(1 for r in clf.roles if r != tc.OTHER) == 2
    team_frac = sum(s for r, s in zip(clf.roles, clf.sizes) if r != tc.OTHER) / sum(clf.sizes)
    assert team_frac > 0.97, (clf.roles, clf.sizes)
    res = None
    for f in range(30):
        res = clf.observe(99, make_crop(rng, KITS["gk_orange"]), 15 + f / 10.0, pitch_xy=(101, 34))
    assert res["is_keeper"] and res["team_id"] == 2, res   # a jobb kaput a 2. csapat védi
    assert clf.is_keeper_track(99, 20.0) and not clf.is_keeper_track(1)
    for f in range(30):
        res = clf.observe(98, make_crop(rng, KITS["ref"], shorts=(20, 20, 20)), 15 + f / 10.0, pitch_xy=(50, 20))
    assert res["is_referee"], res


def test_temporal_voting_with_corrupted_crops():
    rng = np.random.default_rng(8)
    clf = tc.TeamClassifier(grass_lower=GRASS_LO, grass_upper=GRASS_HI)
    for f in range(150):
        t = f / 10.0
        for i in range(10):
            clf.observe(i + 1, make_crop(rng, KITS["red"]), t, pitch_xy=(30, 30))
            clf.observe(i + 11, make_crop(rng, KITS["white"]), t, pitch_xy=(70, 30))
    assert clf.fitted
    raw_ok = voted_ok = n = 0
    for f in range(300):
        t = 15 + f / 10.0
        corrupt = rng.random() < 0.2
        crop = make_crop(rng, KITS["red"], occluder=KITS["white"] if corrupt else None, corrupt=corrupt)
        d, _ = tc.jersey_descriptor(crop, GRASS_LO, GRASS_HI)
        raw_ok += clf._classify(d)[0] == 1
        res = clf.observe(50, crop, t, pitch_xy=(40, 30))
        if f >= 5:
            voted_ok += res["team_id"] == 1
            n += 1
    print(f"  20% hibás kivágás: egyedi besorolás {raw_ok / 300 * 100:.1f}%, szavazással {voted_ok / n * 100:.1f}%")
    assert voted_ok / n > 0.98, voted_ok / n
    assert voted_ok / n > raw_ok / 300


def test_provisional_before_final_fit():
    rng = np.random.default_rng(9)
    clf = tc.TeamClassifier(grass_lower=GRASS_LO, grass_upper=GRASS_HI, fit_seconds=30.0)
    first = clf.observe(1, make_crop(rng, KITS["red"]), 0.0)
    assert first["role"] == "unknown" and first["team_id"] is None
    for f in range(40):
        t = f / 10.0
        for i in range(5):
            clf.observe(i + 1, make_crop(rng, KITS["red"]), t)
            clf.observe(i + 11, make_crop(rng, KITS["white"]), t)
    assert clf.has_model and not clf.fitted
    assert clf.team_colors() is not None


def test_auto_teams_enabled():
    full = dict(team1_lower=[0, 0, 0], team1_upper=[179, 255, 255], team2_lower=[0, 0, 0], team2_upper=[179, 255, 255])
    assert tc.auto_teams_enabled(full)
    assert not tc.auto_teams_enabled(dict(full, auto_teams=False))
    assert not tc.auto_teams_enabled(dict(full, team1_lower=[0, 100, 100], team1_upper=[10, 255, 255]))


def test_observe_cost():
    rng = np.random.default_rng(10)
    clf = tc.TeamClassifier(grass_lower=GRASS_LO, grass_upper=GRASS_HI)
    crops = [make_crop(rng, KITS["red" if i % 2 else "white"]) for i in range(400)]
    for f in range(130):
        for i in range(20):
            clf.observe(i, crops[(f * 20 + i) % 400], f / 10.0)
    assert clf.fitted
    t0 = time.perf_counter()
    for j, c in enumerate(crops):
        clf.observe(j % 20, c, 20 + j / 200.0)
    per = (time.perf_counter() - t0) / len(crops) * 1e3
    t0 = time.perf_counter()
    clf2 = tc.TeamClassifier()
    clf2._X = [tc.jersey_descriptor(c)[0] for c in crops * 4]
    clf2._lab = [np.zeros(3, np.float32)] * len(clf2._X)
    clf2._px = [None] * len(clf2._X)
    clf2._tid = [i % 20 for i in range(len(clf2._X))]
    clf2._fit(final=True)
    fit_ms = (time.perf_counter() - t0) * 1e3
    print(f"  observe (illesztés után): {per:.3f} ms / detektálás; illesztés 1600 mintán: {fit_ms:.1f} ms")
    assert per < 1.5, per


def _veo_overcast_run(seed=11, seconds=40.0, hz=8.0, churn_sec=1.5, with_grass_model=True):
    """
    Valódi Veo-felvétel (borús nappal) mintájára: fehér mez + fekete nadrág vs királykék mez + kék
    nadrág, csupa fekete bíró, piros/rózsaszín kapus; 1080p, 40–120 px magas alakok; a kamera
    egyszerre csak a pálya egy részét látja (kb. 4–6 játékos csapatonként), és a követő azonosítói
    sűrűn cserélődnek (ID-csere churn_sec-enként) – pályakoordináta nincs (sikertelen kalibráció).
    A régi illesztés itt "minden, ami nem fekete" vs "fekete bíró" lett (mindkét csapat egy klaszter).
    """
    rng = np.random.default_rng(seed)
    white, blue = (222, 226, 220), (165, 75, 30)          # BGR, borús fény (enyhén zöldes-szürkés fehér)
    ref, keeper = (28, 28, 30), (120, 80, 215)
    grass_bgr = np.array([45, 105, 70], np.float32)
    gm = None
    if with_grass_model:
        from pitch_filters import GrassModel
        gm = GrassModel()
        fr = np.clip(grass_bgr + rng.normal(0, 6, (540, 960, 3)), 0, 255).astype(np.uint8)
        fr[:150] = 235                                       # fehér égbolt
        for _ in range(3):
            gm.observe(fr)
        gm.flush()
    clf = tc.TeamClassifier(grass_model=gm) if gm is not None else tc.TeamClassifier(grass_lower=GRASS_LO, grass_upper=GRASS_HI)
    people = [dict(kit=white, shorts=(25, 25, 25), team="A", role="player") for _ in range(6)]
    people += [dict(kit=blue, shorts=(150, 60, 25), team="B", role="player") for _ in range(6)]
    people += [dict(kit=ref, shorts=(25, 25, 25), team=None, role="referee"),
               dict(kit=keeper, shorts=(110, 70, 200), team="B", role="keeper")]
    next_id = [1000]
    for p in people:
        p["tid"], p["t_id"] = next_id[0], 0.0
        next_id[0] += 1
    recs = []
    for f in range(int(seconds * hz)):
        t = f / hz
        for i, p in enumerate(people):
            # a kamera az alakok kb. felét látja (a kapus ritkábban)
            vis_p = 0.25 if p["role"] == "keeper" else 0.6
            if rng.random() > vis_p:
                continue
            if t - p["t_id"] >= churn_sec:                    # ID-csere
                p["tid"], p["t_id"] = next_id[0], t
                next_id[0] += 1
            crop = make_crop(rng, p["kit"], h=int(rng.integers(40, 121)), shorts=p["shorts"])
            res = clf.observe(p["tid"], crop, t, pitch_xy=None)
            recs.append((p["role"], p["team"], res, t, p["tid"]))
    return clf, recs


def test_veo_overcast_white_vs_blue_with_black_referee():
    for gm in (True, False):
        clf, recs = _veo_overcast_run(with_grass_model=gm)
        assert clf.fitted
        after = clf.fit_seconds + 2.0
        acc = max(_accuracy(recs, after, {"A": 1, "B": 2}), _accuracy(recs, after, {"A": 2, "B": 1}))
        ref_in_team = [res["team_id"] is not None for role, _, res, t, _ in recs if role == "referee" and t > after]
        cols = clf.team_colors()
        print(f"  Veo borús (fűmodell={'igen' if gm else 'nem'}): pontosság={acc * 100:.1f}%, bíró csapatban: "
              f"{np.mean(ref_in_team) * 100:.0f}%\n    {clf.describe()}")
        assert acc > 0.9, acc
        assert np.mean(ref_in_team) < 0.2, np.mean(ref_in_team)
        # a fehér mez saját klaszter (nem nyelte el a fűszűrés / a fénycsúcs-vágás)
        assert max(min(cols["team1"]), min(cols["team2"])) > 150, cols
        # a fekete bíró nem csapat
        assert max(max(cols["team1"]), max(cols["team2"])) > 120 and min(max(cols["team1"]), max(cols["team2"])) > 80, cols


if __name__ == "__main__":
    tests =[(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
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
