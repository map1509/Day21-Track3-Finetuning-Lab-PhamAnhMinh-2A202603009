"""Package real NB1-NB4 artifacts and code for full Colab NB5 evaluation."""
import hashlib
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def build():
    paths = [ROOT / "requirements.txt", ROOT / "data/checksums.json"]
    for folder, pattern in (("src", "*.py"), ("notebooks", "*.py"),
                            ("scripts", "*.py"), ("tests", "*.py"),
                            ("data", "*.jsonl")):
        paths.extend((ROOT / folder).rglob(pattern))
    paths.extend(p for p in (ROOT / "results").glob("*") if p.is_file() and p.suffix in {".json", ".csv", ".md"})
    for key in ("correct", "attn_only", "wrong_lr", "qlora"):
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            path = ROOT / "adapters" / key / name
            assert path.is_file(), f"Missing {path}"
            paths.append(path)
    package = ROOT / "colab/NB5_INPUT.zip"
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            archive.write(path, path.relative_to(ROOT).as_posix())
    digest = hashlib.sha256(package.read_bytes()).hexdigest()
    setup = f'''import hashlib, os, pathlib, subprocess, sys, zipfile
import torch
from google.colab import files
assert torch.cuda.is_available(), "Select T4 GPU before running"
print("GPU:", torch.cuda.get_device_name(0))
print("Choose NB5_INPUT.zip from your local colab folder")
uploaded = files.upload()
assert "NB5_INPUT.zip" in uploaded, "Upload the generated NB5_INPUT.zip"
payload = uploaded["NB5_INPUT.zip"]
assert hashlib.sha256(payload).hexdigest() == {digest!r}, "Wrong input package"
root = pathlib.Path("/content/lab21_nb5_portable")
assert not (root / "results/verdict.json").exists(), "Already evaluated: download results before another run"
root.mkdir(exist_ok=True)
with zipfile.ZipFile("NB5_INPUT.zip") as archive:
    archive.extractall(root)
del uploaded, payload
os.chdir(root)
os.environ["COMPUTE_TIER"] = "T4"
os.environ.pop("BASE_MODEL", None)
os.environ.pop("EVAL_LIMIT", None)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
'''
    run = '''import csv, json, subprocess, sys, pathlib
root = pathlib.Path.cwd()
with (root / "results/runs.csv").open() as fh:
    rows = {r["run"]: r for r in csv.DictReader(fh)}
for key in ("correct", "attn_only", "wrong_lr", "qlora"):
    assert int(rows[key]["max_steps"]) == int(rows[key]["actual_steps"]) == 30
subprocess.run([sys.executable, "-u", "scripts/colab_run.py", "nb5"], check=True)
'''
    download = '''import json, pathlib, shutil
from google.colab import files
root = pathlib.Path.cwd()
verdict = json.loads((root / "results/verdict.json").read_text())
ranking = json.loads((root / "results/target_ranking.json").read_text())
qualitative = json.loads((root / "results/qualitative_selected.json").read_text())
assert len(ranking) == 4, "Not all adapters evaluated"
assert len(qualitative["examples"]) >= 5
print(json.dumps(verdict, ensure_ascii=False, indent=2))
print(json.dumps(ranking, ensure_ascii=False, indent=2))
print("Real FT losses vs baseline (b):", qualitative["n_losses"])
shutil.make_archive("nb5_results", "zip", "results")
files.download("nb5_results.zip")
print("Evaluation complete. PASS and FAIL are both valid measured outcomes.")
'''
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [
        "# NB5 — full evaluation of four trained adapters\n",
        "Select T4 GPU and Run all. In Setup, upload NB5_INPUT.zip generated beside this notebook.\n",
        "Uses frozen baseline target=0.84 and full eval. No training or baseline reruns.\n",
        "The verdict covers correct on target/regression/format/latency; contrasts are ranked by target.\n",
        "Final cell downloads nb5_results.zip. If interrupted after verdict, download partial results before resetting runtime.\n"]}]
    for source in (setup, run, download):
        compile(source, "<nb5-cell>", "exec")
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": source.splitlines(True)})
    for index, cell in enumerate(cells):
        cell["id"] = f"nb5-portable-{index}"
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"accelerator": "GPU", "kernelspec": {
                    "name": "python3", "display_name": "Python 3"}}}
    out = ROOT / "colab/NB5_LOCAL_READY.ipynb"
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Built {out.name} and {package.name} ({package.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    build()
