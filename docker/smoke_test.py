"""Build both targets, then run python docker/smoke_test.py (no host ML dependencies)."""

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
        description="Exercise API and Streamlit images with a synthetic checkpoint"
    )
    parser.add_argument("--api-image", default="food-classifier-api:local")
    parser.add_argument("--demo-image", default="food-classifier-demo:local")
    options = parser.parse_args()
    name = f"food-api-smoke-{uuid.uuid4().hex[:10]}"
    demo_name = f"food-demo-smoke-{uuid.uuid4().hex[:10]}"
    no_model_name = f"food-api-nomodel-smoke-{uuid.uuid4().hex[:10]}"
    with tempfile.TemporaryDirectory(prefix="food-api-smoke-") as directory:
        temporary_root = Path(directory).resolve()
        assert temporary_root.parent == Path(tempfile.gettempdir()).resolve()
        # TemporaryDirectory is 0700 on Linux. Mount a readable child, keeping
        # the host-only parent private while allowing container UID 10001 access.
        root = temporary_root / "models"
        root.mkdir()
        root.chmod(0o755)
        script = (
            "import torch; from pathlib import Path; from PIL import Image; "
            "from src.models import build_model; from src.data.constants import CLASSES; "
            "assert torch.version.cuda is None; assert not Path('/app/.env').exists(); "
            "torch.save({'model_state_dict':build_model().state_dict(),"
            "'metadata':{'image_size':128,'normalization':'food11','classes':list(CLASSES)}},"
            "'/models/best.pt'); Image.new('RGB',(80,60),'orange').save('/models/photo.png')"
        )
        docker(
            "run",
            "--rm",
            "--user",
            "0:0",
            "--mount",
            f"type=bind,source={root},target=/models",
            options.api_image,
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
                "--mount",
                f"type=bind,source={root},target=/models,readonly",
                options.api_image,
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

            # Without a mounted checkpoint the API stays up, reports why and refuses to predict;
            # the container healthcheck then fails, so `compose up --wait` does not succeed.
            docker("run", "-d", "--name", no_model_name, options.api_image)
            no_model_check = """
import io, json, time, urllib.error, urllib.request
from PIL import Image
deadline = time.monotonic() + 60
while True:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
            health = json.load(response)
        break
    except OSError:
        if time.monotonic() > deadline:
            raise
        time.sleep(0.5)
assert health["model_loaded"] is False and "not found" in health["error"], health
photo = io.BytesIO()
Image.new("RGB", (32, 32), "orange").save(photo, format="PNG")
boundary = "no-model-boundary"
body = (
    f'--{boundary}\\r\\nContent-Disposition: form-data; name="file"; filename="p.png"\\r\\n'
    "Content-Type: image/png\\r\\n\\r\\n"
).encode() + photo.getvalue() + f"\\r\\n--{boundary}--\\r\\n".encode()
request = urllib.request.Request(
    "http://127.0.0.1:8000/predict",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)
try:
    urllib.request.urlopen(request, timeout=5)
    raise AssertionError("predict should fail without a model")
except urllib.error.HTTPError as error:
    assert error.code == 503, error.code
"""
            docker("exec", no_model_name, "python", "-c", no_model_check)
            unhealthy = docker(
                "exec", no_model_name, "python", "docker/healthcheck.py", check=False
            )
            assert unhealthy.returncode != 0
            docker(
                "run",
                "-d",
                "--name",
                demo_name,
                "--publish",
                "127.0.0.1::8501",
                "--env",
                "CHECKPOINT_DIR=/models",
                "--mount",
                f"type=bind,source={root},target=/models,readonly",
                options.demo_image,
            )
            demo_port = docker("port", demo_name, "8501/tcp").stdout.strip().split(":")[-1]
            deadline = time.monotonic() + 60
            while True:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{demo_port}/_stcore/health", timeout=2
                    ) as response:
                        if response.read().strip() == b"ok":
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("Demo did not become ready within 60 seconds")
                time.sleep(0.5)
            ui_script = """
from pathlib import Path
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app/ui/streamlit_app.py", default_timeout=60).run()
assert not at.exception
assert at.selectbox[0].options == ["/models/best.pt"], at.selectbox[0].options
# Input size and normalisation come from the checkpoint metadata, not from the UI.
assert any("128 px" in caption.value for caption in at.sidebar.caption)
at.file_uploader[0].upload("photo.png", Path("/models/photo.png").read_bytes(), "image/png").run()
at.button[0].click().run()
assert not at.exception and not at.error
assert len(at.metric) == 1
at.radio[0].set_value(5).run()
assert not at.exception and len(at.metric) == 1
print("Real Streamlit prediction and Top-5 rerun passed")
"""
            result = docker("exec", demo_name, "python", "-c", ui_script)
            print(result.stdout.strip())
            docker("exec", demo_name, "python", "docker/healthcheck.py", "demo")
            docker("exec", demo_name, "python", "-c", "import os; assert os.getuid() != 0")
            print(
                "Both container smokes passed: CPU weights, readiness, API upload, "
                "demo prediction, non-root and missing-checkpoint handling"
            )
        except Exception:
            for container in (name, no_model_name, demo_name):
                logs = docker("logs", container, check=False)
                print(f"--- {container}\n{logs.stdout}\n{logs.stderr}")
            raise
        finally:
            for container in (name, no_model_name, demo_name):
                docker("rm", "-f", container, check=False)


if __name__ == "__main__":
    main()
