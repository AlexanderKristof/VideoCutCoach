# Edzői igények és fejlesztési terv (egyeztetés: 2026-11-27)

## Kinek, mire
* **Eladásra, más edzőknek.** Hosszú távon: Veo, egyéb AI-kamerák (Pixellot, Trace, XbotGo …), telefon /
  állványos kamera, kapu mögötti kamera; felnőtt amatőr, utánpótlás nagypályán, kispálya / gyerek, futsal.
* **Első eladható változat: Veo + nagypálya, Kapus mód.** A többi kamera és pályatípus utána.
* Kapu mögötti kamera: most nincs felvétel, később lesz.
* Vásárlói gép: **laptop is menjen** (videókártya nélkül lassabban, GPU-val gyorsan); felhő nem (a videók az
  edző gépén maradnak).
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
  Ha nem talál biztosan kapust, **mind a három mód** elérhető: több jelölt képe (választás), rákattintás egy
  képkockán, kapusmez színének megadása palettáról.
* **Csak a kapusos jelenetekkel dolgozik** (gyorsítás). **Kapusos jelenet szabálya** (egyeztetve):
  1. **minden, amiben a kapus a labdához ér – bárhol a pályán** (felszabadítás, söprögetés a félpályán is);
  2. amikor **a labda a kapus közelében** van;
  3. **a végkimenet alapján visszafelé:** ha a támadás góllal, lövéssel vagy a 16-oson belülre érkező labdával
     (beadás, szabadrúgás az ellenfél térfeléről, mélységi labda) végződött, a támadás eleje is bekerül –
     **nem számít, mozdult-e a kapus**; az edző dönti el, jól döntött-e (ment / maradt);
  4. **inkább több, „?” jellel, mint kevesebb.**
  * A > 2 m mozgás NEM feltétel (lehet, hogy éppen a nem-mozdulás a lényeg).
  * Ritka, eredmény nélküli helyzet (pl. 40 m-es lövőcsel, jól reagált): **az edző veszi fel kézzel**
    („➕ Jelenet ide”); ha gyakran felvesz ilyet, a program megtanulja és ajánlja.
* **Labdakihozatal** (a kapustól VAGY a védőktől indul – kapus módba akkor, ha a kapus is részt vesz a
  passzláncban). Kimenet:
  * **sikeres:** passzokkal legalább a félpályáig / az ellenfél térfeléig jut;
  * **labdatartás:** nálunk marad, de nem jut előre – a jelenet addig tart, amíg el nem jut a félpályáig vagy
    el nem veszti (nincs időkorlát);
  * **labdavesztés:** az ellenfélnél köt ki → még pár mp (kb. 5) a reakcióig (visszatámadás / védekezés);
    ha helyzet / lövés lesz belőle, addig.
* **Jelenet-ablak: típusonként más** (pl. lövés 6 mp előtte + 3 utána; kirúgás 2 + 6), beállítható.
* **Kalapok:** védések + kapott gólok; beadások, szögletek; labdakihozatal; kifutások, 1 az 1 elleni; egyéb.
* **Bizonytalan jelenet: bekerül, „?” jellel** (inkább egy felesleges, mint egy kimaradt).
* **Menet: elemzés → jelenetlista → az edző átnézi (✓ / ✗ / ⭐, törlés, új) → vágás.** A jelölésekből a
  program tanul.
* **Mozgás-adatok** (hol állt lövéskor, mennyit mozdult előtte, követte-e a labdát): **opciók**, alapból csak
  a videó.
* **Kimenet kapusonként:** válogatás kalapokba; egy hosszú videó időrendben; kapus-statisztika
  (**opció, bepipálható**): védések / kapott gólok, beadások kezelése, labdakihozatal (rövid / hosszú, kézzel /
  lábbal; sikeres / labdatartás / labdavesztés), kifutások + távolságok / hőtérkép.
* A jelenetek között nincs átmenet (egymás után); az eredeti hang marad, opcióként kikapcsolható.

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
* Bontás: **félidőnként**, **csapatonként egymás mellett**, **pályatérképen** (hol történtek).
* Riport: a programban egy lapon (kattintható: a számra kattintva lejátssza a jeleneteket), PDF, Excel / CSV.
  PDF: fejléc (csapatok, dátum, eredmény, logó), számok táblázatban, pályarajzok (lövés-, labdavesztés-,
  hőtérkép), képkockák a ⭐ jelenetekből az edző rajzaival.
* Több meccs összesítése (szezon, fejlődés): később.
* Nevek: csapatnevek (+ kapusok neve); játékosonkénti azonosítás (mezszám) később.

## Meccsadatok, csapatok, mentés
* Meccsadatok (csapatok, dátum, korosztály / pályatípus, félidőhossz): **utólag, ha kell** – az elemzés azonnal
  indulhat.
* **Csapatprofilok lenyíló listából:** a program rákérdez, és elmenti az eddigi csapatokat (név, logó, mezszín,
  kapusok), legközelebb csak ki kell választani.
* Eredmények mentése: **mindig megkérdezi**, hova.

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

## Részletek (harmadik kérdéskör)
### Kapus mód felülete
* Jelenetlista: **kártyák kis képpel** (típus, meccsidő, ✓ / ✗ / ⭐ / „?”), kattintásra lejátszás.
* Szűrés: **kalap szerint**.
* Kapus kijelölése: **gyors előkeresés után, az elemzés elején** (1–2 perc), utána felügyelet nélkül fut.
* Sebesség: **Gyors / Kiegyensúlyozott / Pontos**, mellette becsült idő az adott gépen.

### Szerkesztő
* Kimerevítés alapból **3 mp** (állítható).
* Rajz: szín és vastagság, szaggatott nyíl (futás), terület kiemelése (átlátszó zóna), reflektor.
* Nagyítás: **opció, az edző jelöli ki** a területet.
* Projekt **automatikus mentéssel**, visszavonással (Ctrl+Z), később újra megnyitható és exportálható.

### Export
* Felbontás választható, alap: **eredeti** (1080p / 720p is).
* Telefonos változat: **opció**, az edző választja (Messenger / WhatsApp méretkorlát vagy minőség + link).
* Logó: helye, mérete, átlátszósága **választható**.
* Fájlnév: **Dátum_Csapatok_Tartalom** (pl. 2026-11-28_FCValami-Ellenfel_Kapus1_Vedesek.mp4).

### Tanítás, béta, név
* A nagy címkézés a fejlesztő feladata; a vásárlóknak csak a ✓ / ✗ / ⭐ opció.
* Címkéző csomag: **minden jelölt + kétes + véletlen jelenet**, a **kapusos jelenetek elöl**; a haladás
  mentődik, bármikor félbehagyható és folytatható.
* Felvételek a tanításhoz: saját csapat (klub engedélyével) + nyilvános Veo-meccsek (tanításhoz érdemes a
  klubtól engedélyt kérni, főleg utánpótlásnál).
* Béta-tesztelők (van néhány edző): egyszerű telepítő (setup.exe, Python nélkül), rövid útmutató,
  hibajelentés gomb (napló, videó nélkül), automatikus frissítés.
* Határidő: nincs – inkább legyen jó, mint gyors.
* **Termék neve: VideoCutCoach** (egy helyen átírható legyen).

## Hosszanti (kapu fölötti) kamera – később
* A kapus-elemzéshez a **kapu mögött, magasan (4–6 m+)** álló, a pályát hosszában látó kamera a legjobb
  (szögzárás, fellépés, lábmunka a kapus szemszögéből); álló kamera → egyszeri kalibráció, pontosabb.
* Térfélcsere: a program **a kiválasztott kapust mindkét félidőben követi** (csapat / mezszín alapján).
  Három szint (béta-útmutatóba is):
  * **áthelyezés nélkül, 1080p:** a távoli félidő durva (gól, támadás a kapu előtt, nagy kifutások; „?” jellel,
    bővebb ablakkal, a szerkesztő felajánlja a nagyítást);
  * **áthelyezés nélkül, 4K:** közepes (gól, lövés, kifutás jól; technika nem);
  * **szünetben áthelyezve / két kamera:** pontos mindkét félidőben.
* Ha van Veo-felvétel is: a program szinkronizálja, a távoli félidőhöz azt használja.
* Ajánlott kamera: 4K (vagy jobb), 50 fps, nagy látószög halszem-torzítás nélkül, rögzített (nem követő)
  kép, kikapcsolt képstabilizálás, magas állvány / rúd, a kép közepén a kapu, a vonalak látszódjanak.
