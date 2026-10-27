"""Reserved reproduction command; no headline results exist at Gate 0."""

import sys


def main() -> int:
    """Refuse to present unimplemented experiments as a successful reproduction."""
    print(
        "Headline reproduction is unavailable until T7.2. "
        "Follow TODO.md in order. No results were generated.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
