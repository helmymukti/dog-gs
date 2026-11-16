"""
Authorship, affiliations and citation of DOG-GS (single source used by the GUI, CLI and reports).

Keep in sync with AUTHORS.md and CITATION.cff (tests/test_about.py checks this).
"""
from __future__ import annotations

from . import __version__

NAME = "DOG-GS"
FULL_NAME = "Direct Orthophoto Generation from Georeferenced 3D Gaussian Splatting"
TAGLINE = "Georeferenced 3D Gaussian Splatting to true orthophoto and DSM"
REPO_URL = "https://github.com/helmymukti/dog-gs"
LICENSE = "MIT"
YEAR = 2026

AFFILIATIONS = {
    1: "Department of Geomatics Engineering, Institut Teknologi Sepuluh Nopember (ITS), Surabaya, Indonesia",
    2: "Department of Geomatics Engineering, Universitas Gadjah Mada (UGM), Yogyakarta, Indonesia",
}

# (display name, affiliation key, corresponding author)
AUTHORS = [
    ("Helmy Mukti Wijaya", 1, True),
    ("Hepi Hapsari Handayani", 1, False),
    ("Teguh Hariyanto", 1, False),
    ("Harintaka", 2, False),
]
CORRESPONDENCE = "helmy.mukti.wijaya@gmail.com"

# software and methods DOG-GS builds on
ACKNOWLEDGEMENTS = [
    ("3D Gaussian Splatting", "Kerbl, Kopanas, Leimkühler & Drettakis (2023), Inria / GRAPHDECO", "PLY model format and SH convention"),
    ("GeoSplat", "Wijaya, Handayani, Hariyanto & Harintaka (2026)", "MIT licence — georeferenced Gaussian models"),
    ("tifffile", "C. Gohlke", "BSD licence"),
    ("Numba", "Lam, Pitrou & Seibert (2015)", "BSD licence"),
    ("OpenCV", "OpenCV team", "Apache 2.0 licence"),
    ("Qt for Python (PySide6)", "The Qt Company", "LGPL licence"),
]


def short_names() -> str:
    """'H. M. Wijaya, H. H. Handayani, T. Hariyanto, Harintaka'"""
    out = []
    for name, _, _ in AUTHORS:
        parts = name.split()
        out.append(name if len(parts) == 1 else " ".join(p[0] + "." for p in parts[:-1]) + " " + parts[-1])
    return ", ".join(out)


def credit_line() -> str:
    return f"© {YEAR} {short_names()} · ITS & UGM · {LICENSE} licence"


def citation_apa() -> str:
    names = []
    for name, _, _ in AUTHORS:
        parts = name.split()
        names.append(name if len(parts) == 1 else f"{parts[-1]}, " + " ".join(p[0] + "." for p in parts[:-1]))
    who = ", ".join(names[:-1]) + ", & " + names[-1]
    return f"{who} ({YEAR}). {NAME}: {FULL_NAME} (Version {__version__}) [Computer software]. {REPO_URL}"


def citation_bibtex() -> str:
    authors = " and ".join(
        (n if len(n.split()) == 1 else f"{n.split()[-1]}, {' '.join(n.split()[:-1])}") for n, _, _ in AUTHORS)
    return ("@software{doggs,\n"
            f"  author  = {{{authors}}},\n"
            f"  title   = {{{NAME}: {FULL_NAME}}},\n"
            f"  version = {{{__version__}}},\n"
            f"  year    = {{{YEAR}}},\n"
            f"  url     = {{{REPO_URL}}},\n"
            f"  license = {{{LICENSE}}}\n}}")


def provenance() -> dict:
    """Embedded in every product's metadata so outputs stay attributable."""
    return {"software": NAME, "version": __version__, "repository": REPO_URL, "licence": LICENSE,
            "authors": [{"name": n, "affiliation": AFFILIATIONS[a], **({"corresponding": CORRESPONDENCE} if c else {})}
                        for n, a, c in AUTHORS],
            "cite": citation_apa()}


def banner() -> str:
    lines = [f"{NAME} {__version__} - {TAGLINE}"]
    for n, a, c in AUTHORS:
        lines.append(f"  {n}{'*' if c else ''} ({a})")
    for k, v in AFFILIATIONS.items():
        lines.append(f"  ({k}) {v}")
    lines.append(f"  * {CORRESPONDENCE}   ·   {REPO_URL}")
    return "\n".join(lines)
