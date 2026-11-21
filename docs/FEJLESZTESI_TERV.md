# Focielemző – fejlesztési terv

## Áttekintés

- **Felület:** .NET MAUI (C#, XAML) – csak Windows asztali gépen fut.
- **Elemző motor:** Python (felismerés, követés, eseménydetektálás, vágás).
- **Videóforrás:** főleg Veo felvétel – labdakövető, forgó kamera, egyszerre egy térfelet mutat.
- **Hardver:** NVIDIA RTX 3060 (pontos mód) és integrált videókártyás laptop (gyors mód).

### A két rész kapcsolata

A MAUI program helyi folyamatként vagy helyi szerverként (pl. FastAPI, `localhost`) indítja a Python motort.
- Az elemzés hosszú (percek–órák), ezért a Python folyamatosan jelenti a haladást (pl. JSON sorok: `{"fazis": "kovetes", "szazalek": 42}`).
- Az eredmény egy meccs-JSON: események időbélyeggel, játékospozíciók a 2D pályán, statisztikák.
- A klipek vágását a Python végzi ffmpeg-gel, a MAUI csak listáz, előnézetet mutat és exportál.

---

## Alapötlet: virtuális pálya („tiszta, nyers színű pálya”)

A videó minden képkockájára ráillesztünk egy **ismert méretű pálya-modellt** (vonalak, 16-os, ötös, középkör, kapuk).

1. **Pályapontok felismerése** képkockánként (vonalkereszteződések, 16-os sarkai, középkör) – kulcspont-modell.
2. **Homográfia:** ebből kiszámoljuk, hogy a kép melyik pontja a valódi pálya melyik pontja (méterben).
   - Mivel a Veo kamera forog, ezt **minden képkockára** (vagy néhány képkockánként + simítás) újra kell számolni.
3. **Ami a pálya-poligonon kívül esik, azt eldobjuk** → nézők, edző, kispad, melegítők kiesnek.
4. A játékosokat és a labdát **tiszta színekkel kirajzoljuk egy 2D (felülnézeti) pályára**:
   - A csapat = egy szín, B csapat = másik szín, kapusok külön szín, bíró sárga/fekete, labda fehér.
   - Ez a „radar / minitérkép” nézet – ebből készül a hőtérkép, a pozíciók, a passzhálózat.
5. **3D csak a labdához kell:** a földön gurul vagy a levegőben repül (ívelés vs. lapos passz).
   - Egy kamerából a magasság csak becsülhető: a labda pályájából (parabola) és látszó méretéből.

**Fontos:** a felismerő modell továbbra is az eredeti videóképen dolgozik (azon tanították), a „tiszta színes pálya” az eredmény megjelenítése és a szűrés eszköze.
Kiegészítésként a színek egyszerűsítése (pl. a mezszínek klaszterezése) segíthet a csapatok szétválasztásában.

---

## Fázisok

### 1. Felismerés tisztítása (ez a mostani probléma)
- Focira tanított modell: `játékos / kapus / bíró / labda` osztályok (beállítás: [MODELL_BEALLITAS.md](MODELL_BEALLITAS.md)).
- **Pályamaszk:** csak az a téglalap marad, amelynek **alsó közepe (a láb)** a pályán belül van.
  - Első lépés: zöld szín (HSV) alapú maszk.
  - Később: a virtuális pálya poligonja (lásd fent).
- **Követés** (ByteTrack / BoT-SORT): stabil azonosítók, a sokáig mozdulatlan szélső alakok kiszűrése.
- **Csapatba sorolás mezszín alapján** (KMeans a felsőtest színeire) – ami nem illik egyik csapatba sem: „egyéb” (edző, mellényes csere).

### 2. Labdakövetés
- Kis, gyors objektum → külön labda-modell vagy nagyobb felbontású kivágás a labda várható helye körül.
- Hiányzó képkockák pótlása interpolációval / Kalman-szűrővel.
- A Veo kamera iránya maga is jelzi, merre van a labda – ez segíthet.

### 3. Virtuális pálya, hőtérkép
- Pályapont-modell + homográfia képkockánként, simítással.
- 2D felülnézet, játékos- és csapat-hőtérkép, megtett táv, pozíciók.

### 4. Események
**Mezőny:** gól, lövés (kapura / mellé), szöglet, szabadrúgás, bedobás, szabálytalanság, beadás, egy kiválasztott játékos összes jelenete.

**Kapus fül (külön szabályok):**

| Esemény | Felismerés alapja |
|---|---|
| Védés | lövés a kapu felé → labda a kapus közelében → nem gól |
| Beadás elfogása | labda oldalról a 16-osba → kapus birtokolja |
| Kifutás / 1v1 | kapus gyorsan kimozdul a kapuból egy támadó felé |
| Kirúgás | labda áll az ötösben → hosszú repülés |
| Ívelés | kapusnál a labda → hosszú, magas repülés |
| Rövid passz / kidobás | kapus után rövid, lapos labdaút csapattárshoz |
| Kapott gól | labda a kapuba megy |

### 5. Klipek és felület
- Eseménylista szűrőkkel (típus, csapat, játékos, félidő).
- Kijelölt események klipjei: esemény előtt X mp, utána Y mp (állítható), összefűzés ffmpeg-gel.
- Statisztika oldal: lövések száma, hőtérkép, 2D minitérkép visszajátszás.
- Kapus fül saját eseménylistával és klipexporttal.

### 6. A 90 perc automatikus kivágása
- Kezdőrúgás: mindenki a saját térfelén, labda a középpontban.
- Sípszó felismerése a hangsávból.
- Félidő: a játékosok elhagyják a pályát / tömörülnek a szélen.
- Kimenet: 1. félidő + 2. félidő, a holtidő nélkül.

### 7. Saját tanítóadat
- Pár száz képkocka kijelölése saját Veo-felvételekből (pl. CVAT vagy Roboflow).
- A modell finomhangolása → nagy pontosságjavulás a saját pályákon, fényviszonyokon.

---

## Teljesítmény – két üzemmód

| | Pontos mód (RTX 3060) | Gyors mód (integrált GPU-s laptop) |
|---|---|---|
| Modell | közepes/nagy YOLO | nano/small YOLO |
| Futtatás | CUDA | ONNX + DirectML vagy OpenVINO |
| Elemzett képkockák | sűrűn (pl. minden 2.) | ritkábban (minden 3–5.), köztük követés |
| Várható idő | perctől néhány tízpercig | akár órák – háttérben, haladásjelzővel |

---

## Következő lépés

Amikor a kód felkerül a repóba: az 1. fázissal kezdünk (nézők, edzők, melegítők kiszűrése).
