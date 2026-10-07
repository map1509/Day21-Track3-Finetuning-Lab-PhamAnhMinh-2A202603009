"""Git newline conversion must not disguise real corpus edits."""
import hashlib
import importlib.util
import pathlib


def test_corpus_hash_accepts_crlf_but_detects_content_edits():
    spec = importlib.util.spec_from_file_location(
        "verify_checksums", pathlib.Path(__file__).resolve().parents[1] / "scripts/verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FileBytes:
        def __init__(self, content):
            self.content = content

        def read_bytes(self):
            return self.content

    original = b'{"input":"ticket"}\n'
    expected = hashlib.sha256(original).hexdigest()[:16]
    assert module._sha(FileBytes(original)) == expected
    assert module._sha(FileBytes(original.replace(b"\n", b"\r\n"))) == expected
    assert module._sha(FileBytes(b'{"input":"changed"}\r\n')) != expected
