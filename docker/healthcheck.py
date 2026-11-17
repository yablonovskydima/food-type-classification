"""Check API readiness with the standard library, without curl in the image."""

import json
import sys
import urllib.request


def main(service: str = "api") -> int:
    try:
        url = (
            "http://127.0.0.1:8501/_stcore/health"
            if service == "demo"
            else "http://127.0.0.1:8000/health"
        )
        with urllib.request.urlopen(url, timeout=3) as response:
            if service == "demo":
                return 0 if response.read().strip() == b"ok" else 1
            data = json.load(response)
        return 0 if data.get("status") == "ok" and data.get("model_loaded") is True else 1
    except (OSError, ValueError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "api"))
