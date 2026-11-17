"""Real container smoke test: python docker/smoke_test.py --image food-classifier-api:local."""

import argparse
import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], check=check, capture_output=True, text=True, timeout=120
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise the API image with a synthetic checkpoint"
    )
    parser.add_argument("--image", default="food-classifier-api:local")
    options = parser.parse_args()
    name = f"food-api-smoke-{uuid.uuid4().hex[:10]}"
    with tempfile.TemporaryDirectory(prefix="food-api-smoke-") as directory:
        root = Path(directory).resolve()
        assert root.parent == Path(tempfile.gettempdir()).resolve()
        script = (
            "import torch; from pathlib import Path; from PIL import Image; "
            "from src.models import build_model; from src.data.constants import CLASSES; "
            "assert torch.version.cuda is None; assert not Path('/app/.env').exists(); "
            "torch.save({'model_state_dict':build_model().state_dict(),"
            "'metadata':{'image_size':64,'normalization':'food11','classes':list(CLASSES)}},"
            "'/models/model.pt'); Image.new('RGB',(80,60),'orange').save('/models/photo.png')"
        )
        docker(
            "run",
            "--rm",
            "--user",
            "0:0",
            "--mount",
            f"type=bind,source={root},target=/models",
            options.image,
            "python",
            "-c",
            script,
        )
        try:
            docker(
                "run",
                "-d",
                "--name",
                name,
                "--publish",
                "127.0.0.1::8000",
                "--env",
                "CHECKPOINT_PATH=/models/model.pt",
                "--mount",
                f"type=bind,source={root},target=/models,readonly",
                options.image,
            )
            port = docker("port", name, "8000/tcp").stdout.strip().split(":")[-1]
            base_url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 60
            while True:
                try:
                    with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                        health = json.load(response)
                    if health.get("model_loaded") is True:
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("Container did not become ready within 60 seconds")
                time.sleep(0.5)
            boundary = "food-api-smoke-boundary"
            body = (
                (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                    'filename="photo.png"\r\nContent-Type: image/png\r\n\r\n'
                ).encode()
                + (root / "photo.png").read_bytes()
                + f"\r\n--{boundary}--\r\n".encode()
            )
            request = urllib.request.Request(
                f"{base_url}/predict?top_k=3",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                prediction = json.load(response)
            assert len(prediction["predictions"]) == 3
            assert all(0 <= item["probability"] <= 1 for item in prediction["predictions"])
            docker("exec", name, "python", "docker/healthcheck.py")
            docker("exec", name, "python", "-c", "import os; assert os.getuid() != 0")
            missing = docker("run", "--rm", options.image, check=False)
            assert missing.returncode != 0 and "CHECKPOINT_PATH" in missing.stderr
            print(
                "Container smoke passed: CPU weights, readiness, upload, non-root, "
                "missing-checkpoint failure"
            )
        except Exception:
            print(docker("logs", name, check=False).stdout)
            print(docker("logs", name, check=False).stderr)
            raise
        finally:
            docker("rm", "-f", name, check=False)


if __name__ == "__main__":
    main()
