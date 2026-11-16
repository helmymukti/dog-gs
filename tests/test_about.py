"""Credits stay consistent across the app, AUTHORS.md and CITATION.cff."""
import json
from pathlib import Path

from doggs import __version__, about

ROOT = Path(__file__).resolve().parents[1]


def _read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_authors_match_authors_md_and_citation():
    md, cff = _read("AUTHORS.md"), _read("CITATION.cff")
    for name, aff, _ in about.AUTHORS:
        assert name in md
        assert name.split()[-1] in cff
        assert about.AFFILIATIONS[aff] in md and about.AFFILIATIONS[aff] in cff
    assert about.CORRESPONDENCE in md and about.CORRESPONDENCE in cff
    zen = json.loads(_read(".zenodo.json"))
    assert len(zen["creators"]) == len(about.AUTHORS)


def test_version_consistent():
    assert f'version = "{__version__}"' in _read("pyproject.toml")
    assert f"version: {__version__}" in _read("CITATION.cff")
    assert f"[{__version__}]" in _read("CHANGELOG.md")


def test_citations_and_credit_line():
    apa, bib = about.citation_apa(), about.citation_bibtex()
    assert "Wijaya, H. M." in apa and "Harintaka" in apa and __version__ in apa
    assert "Wijaya, Helmy Mukti and Handayani, Hepi Hapsari and Hariyanto, Teguh and Harintaka" in bib
    assert about.short_names() == "H. M. Wijaya, H. H. Handayani, T. Hariyanto, Harintaka"
    assert about.credit_line().endswith("ITS & UGM · MIT licence")
    prov = about.provenance()
    assert prov["version"] == __version__ and len(prov["authors"]) == 4
    json.dumps(prov)
