"""Unit tests for boilerplate stripping (src/ingest/clean.py).

These guard the two things that matter: the recurring PDF furniture is removed,
and the clinical text wrapped around it is left untouched (a too-greedy cleaner
that ate real content would silently corrupt the corpus).
"""

from __future__ import annotations

from src.ingest.clean import clean_text


def test_strips_ncbi_running_header():
    page = (
        "6/2/26, 2:06 PM Metformin - StatPearls - NCBI Bookshelf\n"
        "Mechanism of Action Metformin lowers blood glucose by decreasing "
        "hepatic glucose production."
    )
    out = clean_text(page)
    assert "StatPearls - NCBI Bookshelf" not in out
    assert "2:06 PM" not in out
    # the clinical sentence survives intact
    assert "Metformin lowers blood glucose by decreasing hepatic glucose production." in out


def test_strips_ncbi_url_and_page_marker():
    page = "Some clinical text. https://www.ncbi.nlm.nih.gov/books/NBK518983/ 1/8 More text."
    out = clean_text(page)
    assert "ncbi.nlm.nih.gov" not in out
    assert "1/8" not in out
    assert "Some clinical text." in out
    assert "More text." in out


def test_strips_ada_journal_footer():
    page = (
        "Pharmacotherapy should be started at diagnosis. "
        "Diabetes Care Volume 49, Supplement 1, January 2026 diabetesjournals.org/care "
        "Choice of glucose-lowering therapy follows."
    )
    out = clean_text(page)
    assert "Supplement 1, January 2026" not in out
    assert "diabetesjournals.org/care" not in out
    assert "Pharmacotherapy should be started at diagnosis." in out
    assert "Choice of glucose-lowering therapy follows." in out


def test_leaves_ordinary_dates_and_fractions_alone():
    # a bare "1/2" or a real date must NOT be mistaken for page furniture
    page = "Give 1/2 tablet. Follow up on 3/15/2026 if symptoms persist."
    assert clean_text(page) == page
