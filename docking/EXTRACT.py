"""
Extract Vina docking scores from output logs or pdbqt results.
Scans directories for files like *_out.txt or *_out.pdbqt and parses affinity lines.
Outputs a CSV with columns: file,set,score
"""
import argparse
import csv
import glob
import os
import re


AFF_RE_TXT = re.compile(r"Affinity:\s*([\-0-9\.]+)")
AFF_RE_PDBQT = re.compile(r"REMARK\s+VINA\s+RESULT:\s*([\-0-9\.]+)")


def parse_score_from_file(path: str):
    score = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = AFF_RE_TXT.search(line)
            if m:
                score = float(m.group(1))
                break
            m2 = AFF_RE_PDBQT.search(line)
            if m2:
                score = float(m2.group(1))
                break
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="Directories to scan")
    ap.add_argument("--out", default="results/docking_scores.csv")
    args = ap.parse_args()

    rows = []
    for d in args.dirs:
        tag = os.path.basename(d.rstrip("/"))
        txts = glob.glob(os.path.join(d, "*_out.txt"))
        pdbqts = glob.glob(os.path.join(d, "*_out.pdbqt"))
        files = txts + pdbqts
        for fp in files:
            s = parse_score_from_file(fp)
            if s is not None:
                rows.append({"file": fp, "set": tag, "score": s})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "set", "score"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out} with {len(rows)} entries")


if __name__ == "__main__":
    main()

