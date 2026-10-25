import json
import sys
from pathlib import Path

from exapt.primekg import audit_primekg


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: audit_primekg.py SOURCE_DIR [OUTPUT_JSON]")

    report = json.dumps(audit_primekg(Path(sys.argv[1])), indent=2) + "\n"

    if len(sys.argv) == 3:
        Path(sys.argv[2]).write_text(report, encoding="utf-8")
    else:
        sys.stdout.write(report)
