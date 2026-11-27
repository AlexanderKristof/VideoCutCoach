# Edzői igények és fejlesztési terv (2026-11-27, egyeztetés alapján)

## Kinek
Eladásra, más edzőknek: idegen csapatok, más Veo-felvételek is működjenek; a beállítás legyen egyszerű.

## Két belépési pont (felül választható, hova importálja a videót)

### 1. Kapus mód (ELSŐ KÖR – ezt csináljuk meg először, pontosan)
* **Kapusok megjelölése az elején:** a program megkeresi a 2 kapust (mezszín, kapu-közelség), megmutat róluk 1–2
  képet, az edző megerősíti vagy rákattint a jóra.
* **Csak a kapusos jelenetekkel dolgozik (±5 mp)** – ez gyorsítja a feldolgozást is.
* **Kapusos jelenet**, ha:
  * a kapus hozzáér a labdához (védés, elkapás, kirúgás, kidobás, passz, kifutás);
  * lövés / beadás a kapujára (akkor is, ha nem ér hozzá);
  * támadás a kapuja előtt (labda kb. 30 m-en belül);
  * a kapus mozgása – **FINOMÍTANDÓ**: nem csak a 2 m-nél nagyobb mozgás számít, hanem az is, ha a helyzet
    mozgást kívánt volna, és a kapus NEM mozdult (pl. lövés / beadás előtti 2 mp-ben állt). Ezt tényként
    rögzítjük (hol állt, mennyit mozdult lövéskor), nem minősítjük – az értékelés az edzőé.
* **Kimenet** (mind kapusonként):
  * válogatás kalapokba (védések, beadások, labdakihozatal, kifutások …);
  * egy hosszú videó időrendben;
  * kapus-statisztika lap – **opció, bepipálható**.
* **Szerkesztő a vágás után (az edző dönt mindenről):** idősáv címkékkel; jelenet törlése / szélének húzása;
  nyíl, kör, vonal rajzolása; kimerevítés (egy képkocka X mp-ig áll, rajta a rajz); szöveges felirat.
  A program nem rajzol magától – csak a kapusjeleneteket vágja ki.

### 2. Fő rész (teljes meccs) – második kör
* Opciók listája; **alapból bepipálva: válogatás-videó + statisztika/riport**; a többi (kapus-elemzés,
  taktikai tábla, hőtérkép …) opció, alapból nincs bepipálva.
* Válogatás: **kalaponként külön fájl ÉS egy összesített** (fejezetekkel), csapatonként szétválasztható.
* Események: mind (passzok saját / ellenfél térfelén, beadás, rögzített helyzetek, lövés, labdaszerzés …).

## Mindenhol
* Időpont: **videóbeli idő ÉS meccsidő** (pl. „🎬 1:12:03 • ⏱ 23:15, 2. félidő”).
* Átnézés / javítás bárhol: asztali programban (✓ / ✗ / ⭐), laptopon offline csomagban; cél a pontosság.
* Nincs véleményező esemény (pl. „helyezkedési hiba”) – csak ami történt.

## Sorrend
1. Holnap (3060-as gép): teljes meccs a mostani programmal → detektor-napló (ebből a kapus mód és a
   hangolás videó nélkül fejleszthető) → címkéző csomag, címkézés elsősorban a kapus-jelenetekre.
2. Kapus mód: kapus-megerősítés, csak kapus-jelenetek (gyorsítás), kalapok, hosszú videó, statisztika (opció),
   „mozgott / nem mozgott lövéskor” adatok.
3. Szerkesztő: idősáv, törlés / vágás, rajz, kimerevítés, felirat → végső videó beégetve.
4. Fő rész: opció-lista, válogatások (külön + összesített), riport.
