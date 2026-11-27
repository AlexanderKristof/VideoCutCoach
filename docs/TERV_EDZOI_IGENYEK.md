# Edzői igények és fejlesztési terv (egyeztetés: 2026-11-27)

## Kinek, mire
* **Eladásra, más edzőknek.** Hosszú távon: Veo, egyéb AI-kamerák (Pixellot, Trace, XbotGo …), telefon /
  állványos kamera, kapu mögötti kamera; felnőtt amatőr, utánpótlás nagypályán, kispálya / gyerek, futsal.
* **Első eladható változat: Veo + nagypálya, Kapus mód.** A többi kamera és pályatípus utána.
* Kapu mögötti kamera: most nincs felvétel, később lesz.
* Vásárlói gép: még nyitott (laptop GPU nélkül / erős PC / felhő) – mindhárom maradjon lehetséges.
* Nyelv: magyar + angol; könnyen fordítható felépítés (további nyelvek később).
* Pályaméret és félidőhossz: **az edző választja** (nagypálya, 9v9, 7v7, futsal / korosztály) – ebből tudja a
  program a méreteket.

## Felület
* **Kezdőképernyő két nagy gombbal: „Kapus elemzés” / „Meccs elemzés”.** A választás után csak az ahhoz
  tartozó lépések látszanak (a mostani fülek helyett).
* Sebesség / pontosság: **opció a programban** (pl. Gyors / Kiegyensúlyozott / Pontos), nem rögzítjük.
* Időpont mindenhol: **videóbeli idő ÉS meccsidő**.
* Nincs véleményező esemény (pl. „helyezkedési hiba”) – csak ami történt.

## Kapus mód (ELSŐ KÖR)
* **Kapusok:** a program megkeresi a 2 kapust, megmutatja, az edző megerősíti / rákattint a jóra.
  **Az edző választ: 1. kapus / 2. kapus / mindkettő** (egy kapus gyorsabb).
  Kapuscsere (pl. félidőben): **egynek számít** („a csapat kapusa”).
* **Csak a kapusos jelenetekkel dolgozik** (gyorsítás). Kapusos jelenet:
  * a kapus hozzáér a labdához (védés, elkapás, kirúgás, kidobás, passz, kifutás);
  * lövés / beadás a kapujára (akkor is, ha nem ér hozzá);
  * támadás a kapuja előtt (labda kb. 30 m-en belül);
  * kapus-mozgás – **finomítandó**: nem csak a > 2 m mozgás, hanem az is, ha a helyzet mozgást kívánt volna,
    és nem mozdult.
* **Jelenet-ablak: típusonként más** (pl. lövés 6 mp előtte + 3 utána; kirúgás 2 + 6), beállítható.
* **Kalapok:** védések + kapott gólok; beadások, szögletek; labdakihozatal; kifutások, 1 az 1 elleni; egyéb.
* **Bizonytalan jelenet: bekerül, „?” jellel** (inkább egy felesleges, mint egy kimaradt).
* **Menet: elemzés → jelenetlista → az edző átnézi (✓ / ✗ / ⭐, törlés, új) → vágás.** A jelölésekből a
  program tanul.
* **Mozgás-adatok** (hol állt lövéskor, mennyit mozdult előtte, követte-e a labdát): **opciók**, alapból csak
  a videó.
* **Kimenet kapusonként:** válogatás kalapokba; egy hosszú videó időrendben; kapus-statisztika
  (**opció, bepipálható**): védések / kapott gólok, beadások kezelése, labdakihozatal (rövid / hosszú, kézzel /
  lábbal, csapatnál maradt-e), kifutások + távolságok / hőtérkép.

## Szerkesztő (a vágás után; minden döntés az edzőé, a program nem rajzol magától)
* Idősáv címkékkel; jelenet törlése / szélének húzása.
* Nyíl, kör, vonal – **csak kimerevített képen** (megáll a kép, rajta a rajz, X mp múlva megy tovább).
* Szöveges felirat; lassítás; nagyítás (a Veo képe széles); ugyanaz a jelenet kétszer (normál + lassítva /
  rajzzal); hangos megjegyzés (mikrofon).
* **Erőforrás-takarékos, minőségromlás nélkül:** szerkesztés közben kis előnézeti kép, a változtatások csak
  utasítások; a végleges videó egyszer, eredeti minőségben készül. Asztali programban, később böngészőben is.
* Feliratok: alapból tiszta videó; **opció: klub logója** a sarokban.
* Kimenet: MP4 a gépre; telefonra optimalizált (Messenger / WhatsApp); feltöltés YouTube / Drive (később).

## Meccs elemzés (MÁSODIK KÖR)
* Opciók; **alapból bepipálva: válogatás-videó + statisztika / riport**; a többi (kapus, taktikai tábla,
  hőtérkép) opció.
* Válogatás: **kalaponként külön fájl ÉS egy összesített** (fejezetekkel), csapatonként.
* Statisztika alapból: passzok térfelenként (sikeres / sikertelen, előre adott); befejezések; rögzített
  helyzetek és kimenetük; labdavesztés / -szerzés pályatérképen.
* Riport: a programban egy lapon (kattintható: a számra kattintva lejátssza a jeleneteket), PDF, Excel / CSV.
* Nevek: csapatnevek (+ kapusok neve); játékosonkénti azonosítás (mezszám) később.

## Adatok és tanulás
* **A videók csak az edző gépén maradnak**; csak az edző tölti fel, ha akarja.
* Tanulás: **helyben, minden edzőnél külön + önkéntes megosztás** (csak a jelölések, videó nélkül), hogy a
  következő frissítés mindenkinek jobb legyen.

## Sorrend
1. **Holnap (3060-as gép):** `git pull videocut main`, teljes meccs a mostani programmal → detektor-napló
   (ebből a kapus mód videó nélkül fejleszthető) → címkéző csomag, címkézés főleg a kapus-jelenetekre.
2. **Kapus mód:** kezdőképernyő, kapus-kiválasztás + megerősítés, csak kapus-jelenetek, típusonkénti ablakok,
   kalapok, „?” jelölés, lista-átnézés, válogatás + hosszú videó, statisztika (opció).
3. **Szerkesztő:** idősáv, törlés / vágás, kimerevítés + rajz, felirat, lassítás, nagyítás, ismétlés, hang,
   logó, telefonos export.
4. **Meccs elemzés:** opció-lista, válogatások (külön + összesített), riport (lap, PDF, CSV).
5. Később: angol nyelv, más kamerák (kapu mögötti, telefon), kisebb pályák / futsal, felhő vagy laptop-mód,
   feltöltés, mezszám.
