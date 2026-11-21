"""
Automatikus csapatba sorolás mezszín alapján (kézi HSV-tartományok nélkül).

Menete:
1. Mezjellemző detektálásonként (jersey_descriptor): a doboz felsőtestéből
   (magasság 20–55%, középső 60% szélesség) fix méretű (12x16) foltot veszünk,
   kihagyjuk a fűszínű pixeleket, a világosság (L) szélső 10–10%-át (árnyék,
   csillanás) levágjuk, és Lab mediánt + kis, telítettséggel súlyozott
   színezet-hisztogramot számolunk. Nincs detektálásonkénti KMeans.
2. Bemelegítés: az első `fit_seconds` másodpercben (vagy `max_fit_samples`
   mintáig) követésenként 0.2 mp-enként gyűjtünk mintát, majd cv2.kmeans k=5
   klaszterre bont. A túl közeli klasztereket (a középpontok távolsága kisebb,
   mint a szórásuk összege) összevonjuk – így ha az első másodpercekben nem
   látszik kapus/bíró, a csapat nem esik szét több klaszterre. A két
   legnagyobb klaszter a két csapat, a többi kicsi klaszter „egyéb” (bíró,
   kapus). A bemelegítés alatt is van ideiglenes illesztés (100 mintánként).
3. Online besorolás: minden követés (track_id) exponenciálisan felejtő
   szavazást tart (felezési idő ~1 mp) a {1, 2, egyéb} címkékre, hiszterézissel,
   így egy-egy rossz kivágás (takarás, elmosódás) nem fordítja át a csapatot.
4. Szerepek: az „egyéb” címkéjű követés kapus, ha a pályakoordinátái többnyire
   a kapu előtti sávban (x < 20 m vagy x > 85 m) vannak (vagy a modell kapus
   osztályt ad); különben bíró. Pályakoordináta nélkül csak „egyéb”.
   A kapus csapata az, amelyik azt a kaput védi (a csapatok átlagos x-e alapján).
5. Csapatsorrend (determinisztikus): 1. csapat az, amelyiknek a játékosai a
   bemelegítés alatt átlagosan a bal térfélen (kisebb x) voltak; ha nincs elég
   pályakoordináta, a nagyobb klaszter. `swap_teams=True` megcseréli.

Csak numpy + OpenCV – torch/ultralytics nélkül tesztelhető.
"""
import math

import cv2
import numpy as np

from pitch_filters import is_full_range

PITCH_LENGTH = 105.0

TORSO_Y = (0.20, 0.55)
TORSO_X = (0.20, 0.80)
PATCH_SIZE = (12, 16)          # (szélesség, magasság) – a felsőtest-folt fix mérete

# A leíró súlyai: a világosság kevésbé számít (árnyék/napfény), a színesség jobban
W_L = 0.6
W_AB = 1.0 / 64.0
W_HIST = 0.5
HUE_BINS = 8
DESC_DIM = 3 + HUE_BINS + 1

OTHER = "other"


def auto_teams_enabled(config):
    """Automatikus mód: ha be van kapcsolva és a kliens nem adott meg csapat-HSV tartományt
    (mindkettő a teljes tartomány = 'nincs beállítva'). A kézi tartomány mindig nyer."""
    if not config.get("auto_teams", True):
        return False
    for k in ("team1", "team2"):
        lo, hi = config.get(f"{k}_lower"), config.get(f"{k}_upper")
        if lo is None or hi is None:
            continue
        if not is_full_range(np.array(lo), np.array(hi)):
            return False
    return True


def _lab_to_rgb(lab):
    px = np.uint8(np.clip(np.round(lab), 0, 255)).reshape(1, 1, 3)
    b, g, r = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0]
    return [int(r), int(g), int(b)]


def jersey_descriptor(crop_bgr, grass_lower=None, grass_upper=None):
    """
    Mezszín-leíró a játékos dobozának kivágásából.
    Visszatér: (leíró float32[DESC_DIM], Lab medián float32[3] (OpenCV 8 bites skála))
    vagy (None, None), ha a kivágás használhatatlan.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None, None
    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 3:
        return None, None
    y1, y2 = int(h * TORSO_Y[0]), max(int(h * TORSO_Y[0]) + 1, int(round(h * TORSO_Y[1])))
    x1, x2 = int(w * TORSO_X[0]), max(int(w * TORSO_X[0]) + 1, int(round(w * TORSO_X[1])))
    torso = crop_bgr[y1:y2, x1:x2]
    if torso.size == 0:
        return None, None
    patch = cv2.resize(torso, PATCH_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    n = hsv.shape[0]

    keep = np.ones(n, dtype=bool)
    if grass_lower is not None and grass_upper is not None and \
            not is_full_range(np.asarray(grass_lower), np.asarray(grass_upper)):
        lo, hi = np.asarray(grass_lower), np.asarray(grass_upper)
        grass = np.all((hsv >= lo) & (hsv <= hi), axis=1)
        keep = ~grass
        # Ha szinte minden "fű": valószínűleg zöld mez (pl. kapus) – ilyenkor mindent megtartunk
        if keep.sum() < 0.25 * n:
            keep[:] = True

    # Árnyék / csillanás: a világosság szélső 10–10%-a kimarad (a fekete/fehér mez megmarad,
    # mert relatív, nem abszolút küszöb)
    L = lab[keep, 0]
    if L.size >= 10:
        Ls = np.sort(L)
        lo_l, hi_l = Ls[int(0.1 * (Ls.size - 1))], Ls[int(0.9 * (Ls.size - 1))]
        trimmed = keep & (lab[:, 0] >= lo_l) & (lab[:, 0] <= hi_l)
        if trimmed.sum() >= 6:
            keep = trimmed

    sel_lab = lab[keep].astype(np.float32)
    sel_hsv = hsv[keep]
    med = np.median(sel_lab, axis=0)

    hh = sel_hsv[:, 0].astype(np.int32)
    s = sel_hsv[:, 1].astype(np.float32)
    v = sel_hsv[:, 2]
    chrom = (s >= 50) & (v >= 40)
    # A piros a 0/180 határon van – eltolással egy rekeszbe kerül
    hbin = (((hh + 11) % 180) * HUE_BINS) // 180
    hist_c = np.bincount(hbin[chrom], weights=0.5 + 0.5 * s[chrom] / 255.0, minlength=HUE_BINS)
    # a színtelen (fehér/szürke/fekete) pixelek egy rekeszbe: a világosságot az L medián hordozza,
    # így a napos/árnyékos fehér mez nem kerül két külön rekeszbe
    hist = np.concatenate([hist_c, [float(np.count_nonzero(~chrom))]])
    tot = hist.sum()
    if tot > 0:
        hist /= tot

    desc = np.empty(DESC_DIM, dtype=np.float32)
    desc[0] = med[0] / 255.0 * W_L
    desc[1] = (med[1] - 128.0) * W_AB
    desc[2] = (med[2] - 128.0) * W_AB
    desc[3:] = hist * W_HIST
    return desc, med


class _Track:
    __slots__ = ("votes", "t", "label", "last_sample_t", "n_pitch", "n_goal", "ex",
                 "n_obs", "n_model_keeper", "last_seen", "keeper")

    def __init__(self, t):
        self.votes = {1: 0.0, 2: 0.0, OTHER: 0.0}
        self.t = t
        self.label = None
        self.last_sample_t = -1e9
        self.n_pitch = 0
        self.n_goal = 0
        self.ex = None
        self.n_obs = 0
        self.n_model_keeper = 0
        self.last_seen = t
        self.keeper = False



class TeamClassifier:
    def __init__(self, grass_lower=None, grass_upper=None, swap_teams=False,
                 fit_seconds=12.0, max_fit_samples=1500, min_fit_samples=60,
                 refit_step=150, sample_interval_sec=0.2, k=7,
                 merge_ratio=0.8, min_team_tracks=3.5, min_part_frac=0.04, outlier_z=4.0, spread_floor=0.04,
                 vote_half_life_sec=1.0, switch_ratio=1.3,
                 goal_zone_m=20.0, keeper_goal_frac=0.6, min_pitch_obs=3,
                 track_ttl_sec=10.0):
        self.grass_lower = grass_lower
        self.grass_upper = grass_upper
        self.swap_teams = bool(swap_teams)
        self.fit_seconds = fit_seconds
        self.max_fit_samples = max_fit_samples
        self.min_fit_samples = min_fit_samples
        self.refit_step = refit_step
        self.sample_interval_sec = sample_interval_sec
        self.k = k
        self.merge_ratio = merge_ratio
        self.min_team_tracks = min_team_tracks
        self.min_part_frac = min_part_frac
        self.outlier_z = outlier_z
        self.spread_floor = spread_floor
        self.vote_half_life_sec = vote_half_life_sec
        self.switch_ratio = switch_ratio
        self.goal_zone_m = goal_zone_m
        self.keeper_goal_frac = keeper_goal_frac
        self.min_pitch_obs = min_pitch_obs
        self.track_ttl_sec = track_ttl_sec

        self._X, self._lab, self._px, self._tid = [], [], [], []
        self._anon = 0             # track_id nélküli minták egyedi (negatív) azonosítója
        self._t0 = None
        self._n_at_fit = 0
        self.fitted = False        # végleges illesztés kész
        self.has_model = False     # van legalább ideiglenes illesztés
        self.centers = None        # (m, D)
        self.spreads = None        # (m,)
        self.roles = []            # klaszterenként: 1, 2 vagy OTHER
        self.colors = []           # klaszterenként RGB
        self.sizes = []
        self.separation = None     # a két csapat középpontjának távolsága / szórásösszeg
        self.order_by = None       # "pitch_x" vagy "size"
        self._team_x = {1: [0.0, 0], 2: [0.0, 0]}   # online átlagos x csapatonként
        self.tracks = {}
        self._last_prune = None

    # ------------------------------------------------------------------ illesztés
    def _cluster(self, X, tids):
        """
        KMeans + csoportosítás. Visszatér: (címkék, [A csapat klaszter-id, B csapat klaszter-id]).
        Heurisztika: egy klaszter akkor "csapatrész", ha sok különböző követésből (játékosból)
        áll (effektív követésszám >= min_team_tracks) és nem túl kicsi; a bíró és a kapus
        klasztere jellemzően 1–2 követés mintáiból áll. A csapatrészeket (pl. napos / árnyékos
        fehér mez) a középpontok távolsága szerint két csoportba vonjuk össze.
        """
        N = len(X)
        k = int(max(2, min(self.k, N // 8)))
        cv2.setRNGSeed(12345)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
        _, labels, _ = cv2.kmeans(X, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel().copy()

        def stats():
            ids = np.unique(labels)
            C = np.array([X[labels == i].mean(axis=0) for i in ids])
            S = np.array([max(self.spread_floor, float(np.sqrt(((X[labels == i] - C[j]) ** 2).sum(1).mean())))
                          for j, i in enumerate(ids)])
            Z = np.array([np.count_nonzero(labels == i) for i in ids])
            return ids, C, S, Z

        def eff_tracks(cid):
            _, cnt = np.unique(tids[labels == cid], return_counts=True)
            p = cnt / cnt.sum()
            return float(np.exp(-(p * np.log(p)).sum()))

        ids, C, S, Z = stats()
        parts = [int(i) for j, i in enumerate(ids)
                 if Z[j] >= self.min_part_frac * N and eff_tracks(i) >= self.min_team_tracks]
        if len(parts) < 2:
            # nincs elég követés-információ: a két legnagyobb klaszter a két csapat
            order = np.argsort(-Z, kind="stable")
            parts = sorted(set(parts) | {int(ids[order[0]]), int(ids[order[1]])},
                           key=lambda i: -Z[list(ids).index(i)])
        # csapatrészek összevonása, amíg két csoport marad (a legközelebbi pár először)
        while len(parts) > 2:
            ids, C, S, Z = stats()
            pos = {int(i): j for j, i in enumerate(ids)}
            best, best_d = None, None
            for a in range(len(parts)):
                for b in range(a + 1, len(parts)):
                    d = float(np.linalg.norm(C[pos[parts[a]]] - C[pos[parts[b]]]))
                    if best_d is None or d < best_d:
                        best, best_d = (a, b), d
            keep, drop = parts[best[0]], parts[best[1]]
            labels[labels == drop] = keep
            parts.remove(drop)
        # a csapathoz nagyon közeli "egyéb" klaszter is a csapathoz tartozik
        ids, C, S, Z = stats()
        pos = {int(i): j for j, i in enumerate(ids)}
        for j, i in enumerate(ids):
            i = int(i)
            if i in parts:
                continue
            r = [float(np.linalg.norm(C[j] - C[pos[t]])) / (S[j] + S[pos[t]]) for t in parts]
            if min(r) < self.merge_ratio:
                labels[labels == i] = parts[int(np.argmin(r))]
        return labels, parts

    def _fit(self, final):
        X = np.array(self._X, dtype=np.float32)
        LAB = np.array(self._lab, dtype=np.float32)
        PX = np.array([np.nan if p is None else p for p in self._px], dtype=np.float64)
        tids = np.array(self._tid, dtype=np.int64)
        labels, parts = self._cluster(X, tids)
        ids = np.unique(labels)
        if len(ids) < 2 or len(parts) < 2:
            return  # nem választható szét – tovább gyűjtünk
        pos = {int(i): j for j, i in enumerate(ids)}
        team_idx = [pos[parts[0]], pos[parts[1]]]
        Z = np.array([np.count_nonzero(labels == i) for i in ids])
        if Z[team_idx[1]] > Z[team_idx[0]]:
            team_idx = team_idx[::-1]          # a nagyobb csapat elöl (méret szerinti sorrendhez)

        # Csapatsorrend: bal térfél (kisebb átlagos x) = 1. csapat, különben a nagyobb klaszter
        mean_x = []
        for j in team_idx:
            xs = PX[(labels == ids[j]) & ~np.isnan(PX)]
            mean_x.append(float(xs.mean()) if xs.size >= 10 else None)
        if mean_x[0] is not None and mean_x[1] is not None:
            if mean_x[1] < mean_x[0]:
                team_idx = team_idx[::-1]
                mean_x = mean_x[::-1]
            self.order_by = "pitch_x"
        else:
            self.order_by = "size"
        if self.swap_teams:
            team_idx = team_idx[::-1]
            mean_x = mean_x[::-1]

        centers, spreads, roles, colors, sizes = [], [], [], [], []
        for j in range(len(ids)):
            m = labels == ids[j]
            c = X[m].mean(axis=0)
            centers.append(c)
            spreads.append(max(self.spread_floor, float(np.sqrt(((X[m] - c) ** 2).sum(1).mean()))))
            if j == team_idx[0]:
                roles.append(1)
            elif j == team_idx[1]:
                roles.append(2)
            else:
                roles.append(OTHER)
            colors.append(_lab_to_rgb(np.median(LAB[m], axis=0)))
            sizes.append(int(m.sum()))
        centers = np.array(centers, dtype=np.float32)

        # Az előző (ideiglenes) illesztéshez képest megfordult-e a sorrend? Ha igen, a szavazatokat is cseréljük
        if self.has_model:
            o1, o2 = self._team_center(1), self._team_center(2)
            n1, n2 = centers[roles.index(1)], centers[roles.index(2)]
            if np.linalg.norm(n1 - o2) + np.linalg.norm(n2 - o1) < np.linalg.norm(n1 - o1) + np.linalg.norm(n2 - o2):
                for tr in self.tracks.values():
                    tr.votes[1], tr.votes[2] = tr.votes[2], tr.votes[1]
                    if tr.label in (1, 2):
                        tr.label = 3 - tr.label
                self._team_x = {1: self._team_x[2], 2: self._team_x[1]}

        self.centers = centers
        self.spreads = np.array(spreads, dtype=np.float32)
        self.roles, self.colors, self.sizes = roles, colors, sizes
        a, b = roles.index(1), roles.index(2)
        self.separation = float(np.linalg.norm(centers[a] - centers[b]) / (spreads[a] + spreads[b])) if a != b else 0.0
        if self.order_by == "pitch_x":
            self._team_x = {1: [mean_x[0] * 10, 10], 2: [mean_x[1] * 10, 10]}
        self.has_model = True
        self.fitted = bool(final)
        self._n_at_fit = len(self._X)
        if final:
            self._X, self._lab, self._px, self._tid = [], [], [], []   # memória

    def _team_center(self, team):
        return self.centers[self.roles.index(team)]

    # ------------------------------------------------------------------ besorolás
    def _classify(self, desc):
        """Egy megfigyelés címkéje: (címke, súly, szín szerinti legközelebbi csapat)."""
        d = np.linalg.norm(self.centers - desc, axis=1)
        z = d / self.spreads
        order = np.argsort(z)
        i = int(order[0])
        t1, t2 = self.roles.index(1), self.roles.index(2)
        color_team = 1 if d[t1] <= d[t2] else 2
        if z[i] > self.outlier_z:
            return OTHER, 0.5, color_team          # egyik klaszterhez sem hasonlít: külön mez
        margin = z[order[1]] / max(z[i], 1e-6) if len(order) > 1 else 10.0
        return self.roles[i], (1.0 if margin > 1.5 else 0.5), color_team

    def left_team(self):
        """Melyik csapat védi a bal (x=0) kaput – az átlagos x alapján, vagy None."""
        (s1, n1), (s2, n2) = self._team_x[1], self._team_x[2]
        if n1 < 5 or n2 < 5:
            return None
        return 1 if s1 / n1 <= s2 / n2 else 2

    def observe(self, track_id, crop_bgr, t, pitch_xy=None, model_keeper=False):
        """
        Egy detektálás feldolgozása. Visszatér:
        {"team_id": 1/2/None, "role": "player"|"keeper"|"referee"|"other"|"unknown",
         "is_keeper": bool, "is_referee": bool, "color_team": 1/2/None, "fitted": bool}
        """
        t = float(t)
        if self._t0 is None:
            self._t0 = t
        self._prune(t)
        desc, lab = jersey_descriptor(crop_bgr, self.grass_lower, self.grass_upper)
        px = None if pitch_xy is None else float(pitch_xy[0])

        tr = None
        if track_id is not None:
            tr = self.tracks.get(track_id)
            if tr is None:
                tr = _Track(t)
                self.tracks[track_id] = tr
            tr.last_seen = t
            tr.n_obs += 1
            tr.n_model_keeper += int(bool(model_keeper))
            if px is not None:
                tr.n_pitch += 1
                tr.n_goal += int(px < self.goal_zone_m or px > PITCH_LENGTH - self.goal_zone_m)
                tr.ex = px if tr.ex is None else 0.9 * tr.ex + 0.1 * px

        # Bemelegítő mintagyűjtés (követésenként legfeljebb 0.2 mp-enként)
        if not self.fitted and desc is not None:
            if tr is None or t - tr.last_sample_t >= self.sample_interval_sec:
                if tr is not None:
                    tr.last_sample_t = t
                self._X.append(desc)
                self._lab.append(lab)
                self._px.append(px)
                if track_id is None:
                    self._anon -= 1
                self._tid.append(int(track_id) if track_id is not None else self._anon)
            n = len(self._X)
            done = (t - self._t0 >= self.fit_seconds and n >= self.min_fit_samples) or n >= self.max_fit_samples
            if done or (n >= self.min_fit_samples and n - self._n_at_fit >= self.refit_step):
                self._fit(final=done)

        if not self.has_model or desc is None:
            label, w, color_team = None, 0.0, None
        else:
            label, w, color_team = self._classify(desc)

        if tr is None:
            final_label = label
        else:
            if label is not None:
                decay = 0.5 ** (max(0.0, t - tr.t) / self.vote_half_life_sec)
                for key in tr.votes:
                    tr.votes[key] *= decay
                tr.votes[label] += w
                tr.t = t
                lead = max(tr.votes, key=tr.votes.get)
                if tr.label is None or (lead != tr.label and
                                        tr.votes[lead] > self.switch_ratio * tr.votes.get(tr.label, 0.0)):
                    tr.label = lead
            final_label = tr.label

        if final_label in (1, 2) and px is not None and label == final_label and w >= 1.0:
            acc = self._team_x[final_label]
            acc[0] += px
            acc[1] += 1

        res = self._resolve(final_label, tr, model_keeper, color_team)
        if tr is not None:
            tr.keeper = res["is_keeper"]
        return res

    def is_keeper_track(self, track_id, t=None):
        """Igaz, ha a követés legutóbb kapusnak minősült (pl. az álló alak szűrő kivételéhez).
        A lekérdezés életben tartja a követést, hogy a szűrt képkockák alatt se felejtsük el."""
        tr = self.tracks.get(track_id)
        if tr is None:
            return False
        if t is not None:
            tr.last_seen = max(tr.last_seen, float(t))
        return tr.keeper

    def _resolve(self, label, tr, model_keeper, color_team):
        res = {"team_id": None, "role": "unknown", "is_keeper": False, "is_referee": False,
               "color_team": color_team, "fitted": self.fitted}
        if label is None:
            return res
        mk = bool(model_keeper) or (tr is not None and tr.n_obs >= 3 and tr.n_model_keeper / tr.n_obs >= 0.3)
        near_goal = (tr is not None and tr.n_pitch >= self.min_pitch_obs
                     and tr.n_goal / tr.n_pitch >= self.keeper_goal_frac)
        if label in (1, 2):
            res.update(team_id=label, role="keeper" if mk else "player", is_keeper=mk)
            return res
        # "egyéb" mez: kapus (a kapu előtt / modell szerint) vagy bíró
        if mk or near_goal:
            team = None
            lt = self.left_team()
            if lt is not None and tr is not None and tr.ex is not None:
                team = lt if tr.ex < PITCH_LENGTH / 2 else 3 - lt
            res.update(team_id=team, role="keeper", is_keeper=True)
        elif tr is not None and tr.n_pitch >= self.min_pitch_obs:
            res.update(role="referee", is_referee=True)
        else:
            res.update(role="other")
        return res

    def _prune(self, t):
        if self._last_prune is not None and t - self._last_prune < 5.0:
            return
        self._last_prune = t
        limit = t - self.track_ttl_sec
        for tid in [k for k, v in self.tracks.items() if v.last_seen < limit]:
            del self.tracks[tid]

    # ------------------------------------------------------------------ kimenet
    def team_colors(self):
        """{"team1": [r,g,b], "team2": [...], "other": [...] vagy None} – None, ha még nincs illesztés."""
        if not self.has_model:
            return None
        others = [(s, c) for r, s, c in zip(self.roles, self.sizes, self.colors) if r == OTHER]
        return {
            "team1": self.colors[self.roles.index(1)],
            "team2": self.colors[self.roles.index(2)],
            "other": max(others)[1] if others else None,
        }

    def describe(self):
        if not self.has_model:
            return "Auto csapat: nincs elég minta az illesztéshez"
        parts = [f"{r}:{c}(n={s})" for r, c, s in zip(self.roles, self.colors, self.sizes)]
        warn = "" if self.separation is None or self.separation >= 1.0 else " – FIGYELEM: a két mez nehezen választható szét"
        return (f"Auto csapat: {'végleges' if self.fitted else 'ideiglenes'} illesztés, sorrend: {self.order_by}, "
                f"szétválás: {self.separation:.2f}{warn}; klaszterek: " + ", ".join(parts))
