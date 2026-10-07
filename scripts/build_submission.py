"""Create Option A ZIP with measured results and the correct adapter only."""
import hashlib
import json
import pathlib
import subprocess
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def build():
    prefix = "lab21_2A202603009"
    paths = [ROOT / "submission/REPORT.md", ROOT / "submission/REFLECTION.md", ROOT / "requirements.txt",
             ROOT / "requirements-cpu.txt", ROOT / "README.md", ROOT / "rubric.md"]
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                             capture_output=True).stdout.decode("utf-8").split("\0")
    # Include tracked supporting files so the repository's structural tests also run.
    paths.extend(ROOT / name for name in tracked if name and (ROOT / name).is_file()
                 and not name.startswith(("submission/", "adapters/", "results/")))
    paths.extend(p for p in (ROOT / "results").iterdir()
                 if p.is_file() and p.suffix in {".json", ".csv", ".md"})
    for folder, pattern in (("notebooks", "*.py"), ("src", "*.py"),
                            ("scripts", "*.py"), ("tests", "*.py"),
                            ("data", "*.jsonl")):
        paths.extend((ROOT / folder).rglob(pattern))
    paths.append(ROOT / "data/checksums.json")
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        paths.append(ROOT / "adapters/correct" / name)
    paths = sorted(set(paths))
    assert all(p.is_file() for p in paths)
    outdir = ROOT / "dist"
    outdir.mkdir(exist_ok=True)
    out = outdir / f"{prefix}.zip"
    manifest = {}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            manifest[relative] = hashlib.sha256(content).hexdigest()
            archive.writestr(f"{prefix}/{relative}", content)
        archive.writestr(f"{prefix}/MANIFEST.json", json.dumps(manifest, indent=2))
        archive.writestr(f"{prefix}/SUBMISSION-NOTES.md", """# Submission — Option A

NB1–NB5 completed; NB6 not performed. Verdict FAIL is the measured regression-gate outcome.
Only the correct adapter is included, as required by Option A. All contrast scores and logs are in results/.
Report includes five target cases and two actual regression losses; no target cases lose to baseline (b).
The source, corpus and tests are included for verification. Run python scripts/verify.py after installing CPU requirements.
Corpus reference hashes normalize Git CRLF/LF checkout differences; NB2 freeze hashes still check exact raw bytes.
MANIFEST.json contains SHA-256 of every included source/artifact file. No .env, credentials, cache or model base weights are included.
""")
    with zipfile.ZipFile(out) as archive:
        assert archive.testzip() is None
        for relative, expected in manifest.items():
            assert hashlib.sha256(archive.read(f"{prefix}/{relative}")).hexdigest() == expected
    print(f"Submission: {out}")
    print(f"Size: {out.stat().st_size / 1024**2:.1f} MiB; {len(paths)} files verified")


if __name__ == "__main__":
    build()
