"""
Az edzői visszajelzés (✓ / ✗ / + hiányzó) összevetése egy eseménylistával – szabálymódosítás után
így látszik, javult-e az eseményfelismerés ugyanazon a meccsen.

    python tools\\eval_feedback.py clips\\feedback_match_events_ab12.json clips\\match_events_ab12.json
    python tools\\eval_feedback.py clips\\feedback_match_events_ab12.json clips\\match_events_UJ.json   (újraelemzés után)
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import feedback as fb  # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(__doc__)
        return 2
    tol = float(argv[2]) if len(argv) > 2 else 3.0
    with open(argv[0], encoding="utf-8") as f:
        feedback = json.load(f)
    with open(argv[1], encoding="utf-8") as f:
        events = json.load(f).get("events", [])
    r = fb.evaluate(feedback, events, tol)
    print(f"Megtalált valódi esemény: {r['talalt']}, kimaradt: {r['kimaradt']}, "
          f"téves (még mindig): {r['teves_maradt']}, téves (már nincs): {r['teves_eltunt']}")
    print(f"Pontosság: {r['pontossag']}, felidézés: {r['felidezes']}  (tűrés: ±{tol:g} mp)")
    for typ, p in sorted(r["tipusonkent"].items()):
        print(f"  {typ:<22} talált {p['talalt']:>3}  kimaradt {p['kimaradt']:>3}  téves {p['teves_maradt']:>3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
