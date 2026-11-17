"""Check API readiness with the standard library, without curl in the image."""

import json
import urllib.request


def main() -> int:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as response:
            data = json.load(response)
        return 0 if data.get("status") == "ok" and data.get("model_loaded") is True else 1
    except (OSError, ValueError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
