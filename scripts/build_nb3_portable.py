"""Build an upload-and-run Colab NB3 with the measured NB2 freeze included."""
import base64
import hashlib
import io
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def build():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from labkit import data
    from labkit.config import OPTIMIZED_PROMPT, get_tier
    frozen = json.loads((ROOT / "results/baselines_frozen.json").read_text(encoding="utf-8"))
    assert frozen["model"] == get_tier("T4").model_id
    assert frozen["baseline_b"]["target"] > frozen["baseline_a"]["target"]
    assert not frozen["smoke_mode"]
    assert hashlib.sha256(OPTIMIZED_PROMPT.encode()).hexdigest()[:16] == frozen["optimized_prompt_sha"]
    for name, expected in frozen["eval_checksums"].items():
        assert hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest() == expected
    paths = [ROOT / "requirements.txt", ROOT / "data/checksums.json"]
    for folder, pattern in (("src", "*.py"), ("notebooks", "*.py"),
                            ("scripts", "*.py"), ("tests", "*.py")):
        paths.extend((ROOT / folder).rglob(pattern))
    paths.extend((ROOT / "data").glob("*.jsonl"))
    paths.extend(ROOT / "results" / name for name in (
        "baselines_frozen.json", "baseline_predictions.json", "mask_proof.json",
        "template_check.json", "token_stats.json", "NB2-MEASURED.md"))
    corpus = [json.loads(line) for line in (ROOT / "data/train_seed.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    train_rows, val_rows = data.split(corpus, train_frac=0.9, seed=42)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            archive.writestr(path.relative_to(ROOT).as_posix(), path.read_bytes())
        for name, rows in (("train", train_rows), ("val", val_rows)):
            archive.writestr(f"data/split/{name}.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    payload = buf.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    setup = f'''import base64, hashlib, io, os, pathlib, subprocess, sys, zipfile
import torch
assert torch.cuda.is_available(), "Select Runtime > Change runtime type > T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
root = pathlib.Path("/content/lab21_nb3_portable")
root.mkdir(exist_ok=True)
assert not (root / "adapters/correct/adapter_config.json").exists(), "Adapter already saved here; run the download cell"
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(root)
os.chdir(root)
os.environ.update(COMPUTE_TIER="T4", MASK_MODE="assistant-only", EPOCHS="2")
os.environ.pop("BASE_MODEL", None)
os.environ.pop("EVAL_LIMIT", None)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
print("Source + frozen NB2 bundle:", {digest!r})
'''
    run = '''import subprocess, sys
subprocess.run([sys.executable, "-u", "scripts/colab_run.py", "nb3"], check=True)
'''
    download = '''import csv, json, pathlib, zipfile
from google.colab import files
root = pathlib.Path.cwd()
adapter = root / "adapters/correct"
assert (adapter / "adapter_config.json").exists(), "Training has not saved the adapter yet"
assert (adapter / "adapter_model.safetensors").exists(), "Adapter weights missing"
with (root / "results/runs.csv").open() as fh:
    rows = list(csv.DictReader(fh))
print(json.dumps(rows, indent=2))
out = root / "nb3_results_and_adapter.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
    for folder in (root / "results", adapter, root / "data/split"):
        for path in folder.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(root))
files.download(str(out))
print("Saved adapter and measured training results. Keep baseline and eval unchanged for NB4/NB5.")
'''
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [
        "# NB3 — train correct LoRA, measured NB2 included\n",
        "Upload this notebook to Colab, select T4 GPU, then Run all. No GitHub push required.\n",
        "Uses frozen baseline (b) target=0.84, full eval and the proved NB1 mask. Does not rerun NB2.\n",
        "Download ZIP includes adapters/correct/, results/ and data/split/. Extract into the repo root.\n"]}]
    for source in (setup, run, download):
        compile(source, "<colab-cell>", "exec")
        cells.append({"cell_type": "code", "execution_count": None,
                      "metadata": {}, "outputs": [], "source": source.splitlines(True)})
    for index, cell in enumerate(cells):
        cell["id"] = f"nb3-portable-{index}"
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"accelerator": "GPU", "kernelspec": {
                    "name": "python3", "display_name": "Python 3"}}}
    out = ROOT / "colab/NB3_LOCAL_READY.ipynb"
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Built {out.name}; frozen NB2 and seed-42 split included, no credentials")
    return out


if __name__ == "__main__":
    build()
