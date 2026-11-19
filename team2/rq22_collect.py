"""Print completed RQ 2.2 summaries as CSV without modifying run files."""
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path("/scratch/gw1b-class/teams/team2/runs")

def main():
    writer = csv.writer(sys.stdout)
    writer.writerow(["run", "n_params", "tokens", "val_loss", "summary_path"])

    for path in sorted(ROOT.glob("team2-rq2.2-*/summary.json")):
        try:
            s = json.loads(path.read_text())
            if s.get("finished") is not True:
                continue
            n, d, loss = s["n_params"], s["tokens"], s["val_loss"]
            if not all(math.isfinite(float(x)) and float(x) > 0
                       for x in (n, d, loss)):
                raise ValueError("Parameters, tokens, and loss must be finite and positive")
            writer.writerow([path.parent.name, n, d, loss, str(path)])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"Cannot read {path}: {exc}", file=sys.stderr)
            return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
