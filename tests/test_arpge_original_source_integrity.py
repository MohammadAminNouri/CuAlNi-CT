"""Guard the three archived main modules against unnoticed source modifications."""

from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "arpge_2021_original"
EXPECTED = {
    "arpge/EBSD/ARPGE.py":
        "93892600f30f2ea2ce817018d5c97ac57a9e50fe9db26594a753e30505d3b077",
    "arpge/Crystallography/CC_Crystals.py":
        "7c8b28e1d49e2493012e646865c1e787fca0dff1d9b169c89f0b5f8036e0d829",
    "arpge/Crystallography/CC_GenOVa.py":
        "9f738fb3451eeb09aafec64753565d0be8731f31ef69784d4f4e77944792c139",
}


def test_original_2021_sources_are_unmodified():
    for name, digest in EXPECTED.items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == digest
