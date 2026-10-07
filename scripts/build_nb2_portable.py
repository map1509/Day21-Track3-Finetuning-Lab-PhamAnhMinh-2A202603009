"""Bundle the current local lab into an upload-and-run Colab notebook, without secrets."""
import base64
import hashlib
import io
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def build():
    paths = [ROOT / "requirements.txt"]
    for folder, pattern in (("src", "*.py"), ("notebooks", "*.py"),
                            ("scripts", "*.py"), ("data", "*.jsonl"),
                            ("tests", "*.py")):
        paths.extend(p for p in (ROOT / folder).rglob(pattern)
                     if "split" not in p.parts)
    paths.append(ROOT / "data/checksums.json")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            # Exclude this generator to avoid recursive notebook bundling.
            if path != pathlib.Path(__file__).resolve():
                archive.writestr(path.relative_to(ROOT).as_posix(), path.read_bytes())
    payload = buf.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    setup = f'''import base64, hashlib, io, os, pathlib, subprocess, sys, zipfile
import torch
assert torch.cuda.is_available(), "Select Runtime > Change runtime type > T4 GPU first"
print("GPU:", torch.cuda.get_device_name(0))
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
root = pathlib.Path("/content/lab21_nb2_portable")
root.mkdir(exist_ok=True)
if (root / "results/baselines_frozen.json").exists():
    raise RuntimeError("Baseline already measured here. Download results before starting another experiment.")
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(root)
os.chdir(root)
os.environ["COMPUTE_TIER"] = "T4"
os.environ.pop("BASE_MODEL", None)
os.environ.pop("EVAL_LIMIT", None)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
print("Local source bundle:", {digest!r})
'''
    run = '''import subprocess, sys
subprocess.run([sys.executable, "-u", "scripts/colab_run.py", "nb1", "nb2"], check=True)
'''
    download = '''import hashlib, json, pathlib, shutil, sys
from google.colab import files
root = pathlib.Path.cwd()
path = root / "results/baselines_frozen.json"
assert path.exists(), "NB2 has not produced real baseline measurements yet"
frozen = json.loads(path.read_text())
print(json.dumps(frozen, ensure_ascii=False, indent=2))
assert frozen["model"] == "unsloth/Qwen3.5-4B"
assert not frozen["smoke_mode"]
for name, count in (("eval_target.jsonl", "n_target"), ("eval_regression.jsonl", "n_regression")):
    assert frozen[count] == len([line for line in (root / "data" / name).read_text().splitlines() if line.strip()])
for name, expected in frozen["eval_checksums"].items():
    assert hashlib.sha256((root / "data" / name).read_bytes()).hexdigest() == expected
sys.path.insert(0, str(root / "src"))
from labkit.config import OPTIMIZED_PROMPT
assert hashlib.sha256(OPTIMIZED_PROMPT.encode()).hexdigest()[:16] == frozen["optimized_prompt_sha"]
a, b = frozen["baseline_a"]["target"], frozen["baseline_b"]["target"]
passed = b > a
summary = f"# NB2 — Measured baseline\\n\\nModel: `{frozen['model']}`\\n\\n| Baseline | Target | Regression | Format | Latency ms |\\n|---|---:|---:|---:|---:|\\n"
for key in ("baseline_a", "baseline_b"):
    s = frozen[key]
    summary += f"| {key} | {s['target']:.4f} | {s['regression']:.4f} | {s['format']:.4f} | {s['latency_ms']:.0f} |\\n"
summary += f"\\n(b) > (a): {passed}. Frozen UTC: {frozen['frozen_at_utc']}.\\n"
(root / "results/NB2-MEASURED.md").write_text(summary, encoding="utf-8")
shutil.make_archive("nb1_nb2_results", "zip", "results")
files.download("nb1_nb2_results.zip")
assert passed, "STOP before training: improve prompt (b) and rerun NB2 in a new experiment"
'''
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [
        "# NB1 + NB2 — local source included\n",
        "Select T4 GPU, then Run all. No GitHub push required. Runs full eval before training.\n",
        "If NB2 fails because (b) <= (a), run the final cell manually to download the evidence.\n",
        "Keep the results ZIP; upload the modified source files for any subsequent training.\n"]}]
    for source in (setup, run, download):
        cells.append({"cell_type": "code", "execution_count": None,
                      "metadata": {}, "outputs": [], "source": source.splitlines(True)})
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"accelerator": "GPU", "kernelspec": {
                    "name": "python3", "display_name": "Python 3"}}}
    for index, cell in enumerate(cells):
        cell["id"] = f"nb2-portable-{index}"
    out = ROOT / "colab/NB2_LOCAL_READY.ipynb"
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Built {out.name}: {len(paths)} source/data files, no .env or credentials")
    return out


if __name__ == "__main__":
    build()
