"""Reserved dataset command; labels require the Gate 1 pricer validation."""

import sys


def main() -> int:
    """Refuse generation until validated pricers and the data pipeline exist."""
    print(
        "Dataset generation is unavailable until T1.4. "
        "Complete pricer validation (T1.1–T1.3) first. No data were generated.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
