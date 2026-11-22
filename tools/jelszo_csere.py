"""
Elfelejtett jelszó: új jelszó beállítása egy helyi fiókhoz (a program saját adatbázisában).

    python tools\\jelszo_csere.py sajat@email.cim
    python tools\\jelszo_csere.py --lista            (a fiókok email címei)

A jelszót kétszer kéri (gépelés közben nem látszik). A fiókok csak ezen a gépen vannak
(tactical_app.db); fejlesztői módban a repó gyökerében, telepített programnál az adatmappában
(TACTICAL_DATA_DIR, alap: %LOCALAPPDATA%\\TacAnaly) – ilyenkor a változót előtte állítsd be.
"""
import argparse
import getpass
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Új jelszó beállítása egy helyi fiókhoz")
    ap.add_argument("email", nargs="?", help="a fiók email címe")
    ap.add_argument("--lista", action="store_true", help="a fiókok email címeinek listája")
    ap.add_argument("--jelszo", help=argparse.SUPPRESS)  # tesztekhez; kézzel ne add meg parancssorban
    a = ap.parse_args(argv)

    # Fejlesztői módban az adatbázis a repó gyökerében van (a motor is onnan fut)
    if not os.environ.get("TACTICAL_DATA_DIR"):
        os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    import app_paths
    from database import SessionLocal, engine
    import models
    from auth import hash_password

    models.Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if a.lista or not a.email:
            users = db.query(models.User).order_by(models.User.created_at).all()
            print(f"Adatbázis: {app_paths.database_url()}")
            if not users:
                print("Nincs még fiók – a programban a Regisztráció gombbal hozhatsz létre.")
            for u in users:
                print(f"  {u.email}")
            return 0 if a.lista else 2

        email = a.email.strip()
        user = db.query(models.User).filter(models.User.email == email).first()
        if user is None:
            print(f"Nincs ilyen fiók: {email}  (a fiókok listája: --lista)")
            return 1
        pw = a.jelszo
        if pw is None:
            pw = getpass.getpass("Új jelszó: ")
            if pw != getpass.getpass("Új jelszó még egyszer: "):
                print("A két jelszó nem egyezik, nem változott semmi.")
                return 1
        if len(pw) < 4:
            print("A jelszó legalább 4 karakter legyen.")
            return 1
        user.hashed_password = hash_password(pw)
        db.commit()
        print(f"Kész: {email} új jelszava beállítva. Most már bejelentkezhetsz vele a programban.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
