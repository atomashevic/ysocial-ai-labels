"""Check rebuilt tables against values printed in the manuscript.

Skipped when the Zenodo data have not been downloaded.
"""

from __future__ import annotations

import pytest

from repro import tables
from repro.paths import RESULTS_DIR

pytestmark = pytest.mark.skipif(not RESULTS_DIR.exists(), reason="run `python -m repro.download` first")


@pytest.fixture(scope="module")
def built() -> dict[str, tables.Table]:
    return {t.key: t for t in tables.build_all()}


def cell(table: tables.Table, row: dict[str, str], column: str) -> str:
    frame = table.frame
    for key, value in row.items():
        frame = frame.loc[frame[key] == value]
    assert len(frame) == 1, row
    return frame.iloc[0][column]


def test_table_1(built):
    t = built["table1"]
    assert cell(t, {"": "Human replies analysed"}, "Visible labels") == "1,286"
    assert cell(t, {"": "Human replies analysed"}, "Hidden labels") == "857"
    assert cell(t, {"": "AI targets among analysed participant replies"}, "Hidden labels") == "35.9%"


def test_table_2(built):
    t = built["table2"]
    assert cell(t, {"Outcome": "Criticism: modelled probability, % [95% CrI]"}, "Visible / AI") == "32.0 [25.9, 38.2]"
    assert cell(t, {"Outcome": "Criticism, primary"}, "Gap difference, pp [95% CrI]") == "13.3 [5.1, 21.4]"
    assert cell(t, {"Outcome": "Downvotes, primary"}, "Gap difference, pp [95% CrI]") == "2.3 [0.3, 4.4]"


def test_rq1_primary_contrast(built):
    row = {"Reference / projection": "Pooled sets (primary)", "Community": "Hidden - visible"}
    assert cell(built["tableS1"], row, "Mean") == "3.5"
    assert cell(built["tableS1"], row, "95% CrI") == "[0.5, 6.5]"


def test_direct_thread_entry_gap(built):
    row = {"Quantity / analysis": "Difference in AI-participant gap"}
    assert cell(built["tableS4"], row, "Mean / count") == "4.4"
    assert cell(built["tableS4"], row, "95% CrI") == "[0.8, 8.1]"


def test_rq2_contact(built):
    frame = built["tableS9"].frame
    visible_received = frame.iloc[3].tolist()
    assert visible_received[1:5] == ["AI to participant", "Visible", "11.3", "[9.4, 13.3]"]


def test_toxicity_thresholds(built):
    row = {"Community": "Visible", "Target": "AI"}
    assert cell(built["tableS23"], row, ">0.5") == "30 (6.9%)"
