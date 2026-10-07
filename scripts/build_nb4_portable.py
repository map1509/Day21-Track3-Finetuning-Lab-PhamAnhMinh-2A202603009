"""Build NB4 with frozen NB2 and NB3 training metadata; download after each contrast."""
import ast
import base64
import hashlib
import io
import json
import zipfile

from build_nb3_portable import ROOT, build as build_nb3


def build():
    nb = json.loads(build_nb3().read_text(encoding="utf-8"))
    setup = "".join(nb["cells"][1]["source"])
    tree = ast.parse(setup)
    payload_call = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "payload" for t in node.targets))
    old_payload = base64.b64decode(ast.literal_eval(payload_call.args[0]))
    old_digest = hashlib.sha256(old_payload).hexdigest()
    buffer = io.BytesIO(old_payload)
    with zipfile.ZipFile(buffer, "a", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("results/runs.csv", (ROOT / "results/runs.csv").read_bytes())
    payload = buffer.getvalue()
    setup = setup.replace(base64.b64encode(old_payload).decode(), base64.b64encode(payload).decode())
    setup = setup.replace(old_digest, hashlib.sha256(payload).hexdigest())
    setup = setup.replace("lab21_nb3_portable", "lab21_nb4_portable")
    setup = setup.replace(
        'assert not (root / "adapters/correct/adapter_config.json").exists(), "Adapter already saved here; run the download cell"',
        'assert not (root / "results/runs.csv").exists(), "Already set up: resume run cells, do not overwrite saved results"')
    nb["cells"] = [{"cell_type": "markdown", "metadata": {}, "source": [
        "# NB4 — three contrasts, same 30 steps as measured correct\n",
        "Select T4 GPU, then Run all. Includes frozen NB2, NB3 runs.csv and seed-42 split.\n",
        "Each run downloads a cumulative ZIP. The correct adapter stays in your local repo.\n",
        "If interrupted with runtime intact, rerun only the unfinished run cell. Saved runs are skipped.\n",
        "If runtime is lost, set up a fresh runtime, upload/extract your last ZIP into /content/lab21_nb4_portable, then resume.\n",
        "Training loss does not decide the winner. NB5 scores all adapters on target.\n"]}]
    def cell(source):
        compile(source, "<nb4-cell>", "exec")
        return {"cell_type": "code", "metadata": {}, "execution_count": None,
                "outputs": [], "source": source.splitlines(True)}
    nb["cells"].append(cell(setup))
    for key in ("attn_only", "wrong_lr", "qlora"):
        nb["cells"].append(cell(f'''# Run {key}; downloads all finished contrasts so far
import os, pathlib, subprocess, sys, zipfile
from google.colab import files
os.chdir("/content/lab21_nb4_portable")
os.environ["ONLY"] = "{key}"
try:
    subprocess.run([sys.executable, "-u", "scripts/colab_run.py", "nb4"], check=True)
finally:
    root = pathlib.Path.cwd()
    out = root / "nb4_after_{key}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in (root / "results", root / "adapters", root / "data/split"):
            if folder.exists():
                for path in folder.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(root))
    files.download(str(out))
'''))
    nb["cells"].append(cell('''import csv, json, pathlib
root = pathlib.Path("/content/lab21_nb4_portable")
with (root / "results/runs.csv").open() as fh:
    rows = {r["run"]: r for r in csv.DictReader(fh)}
for key in ("correct", "attn_only", "wrong_lr", "qlora"):
    assert key in rows, f"Missing result: {key}"
    assert int(rows[key]["actual_steps"]) == int(rows[key]["max_steps"]) == 30
    if key != "correct":
        assert (root / "adapters" / key / "adapter_model.safetensors").exists()
assert abs(int(rows["attn_only"]["trainable_params"]) / int(rows["correct"]["trainable_params"]) - 1) < 0.05
print(json.dumps(rows, ensure_ascii=False, indent=2))
print("NB4 complete. Keep nb4_after_qlora.zip and extract into the local repo root.")
'''))
    for index, c in enumerate(nb["cells"]):
        c["id"] = f"nb4-portable-{index}"
    out = ROOT / "colab/NB4_LOCAL_READY.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Built {out.name}; NB3 metadata included, adapter weights downloaded after each run")
    return out


if __name__ == "__main__":
    build()
