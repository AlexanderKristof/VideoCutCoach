# Előcímkék a 108 valódi képkockához

A gép által javasolt keretek – **neked csak javítanod kell**, nem nulláról kijelölni.

- Készült: YOLOX-m (Apache-2.0) csempés felismeréssel eredeti felbontáson (`tools/prelabel.py --csempe 640`),
  utána a kerítés előtti nézősor automatikusan kiszűrve (a fűterület felső széle alapján).
- Tartalom: 1641 keret – 1517 `player`, 124 `ball` (a labdák egy része téves lehet: fej, pálya-jelölés).
- Mindenki `player`: a **kapusokat** és a **bírókat** neked kell átcímkézni.

## Betöltés CVAT-ba
1. Új feladat (Task): címkék `ball`, `goalkeeper`, `player`, `referee`; képek: a `minta\kep_*.jpg` fájlok.
2. Actions → Upload annotations → **COCO 1.0** → `elocimkek_coco.json`.

## Amit ellenőrizz minden képen
- kapus / bíró átcímkézése (a fekete ruhás a bíró, a zöld / piros mezes a kapus);
- a megmaradt nézők, cserék, edzők törlése (főleg a közeli oldalvonalnál és a kispadnál);
- a labda: téves (fej, folt) törlése, hiányzó (pici, elmosódott) felvétele;
- ha a kerítés közelében egy valódi játékos is kiesett, add hozzá.

A YOLO formátum (`yolo/`) ugyanez szöveges fájlokban (osztály-sorrend: `classes.txt`).
