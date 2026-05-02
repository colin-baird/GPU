"""Snapshot test for the AST-driven signal-flow extractor.

Asserts that the libclang extractor at `tools/diagram_extract_ast.py`
recovers the expected module set and a hand-curated set of high-signal
edges from the timing-model TUs. The full edge set is not snapshotted —
the AST extractor's job is to track the C++ source as it changes, so
the snapshot floor only encodes invariants that should hold for any
correct rendering.

Run via:

    cmake -B build && cmake --build build -j8   # ensure compile_commands.json
    python3 -m pytest tests/test_signal_diagram.py
    # or, if pytest is unavailable:
    python3 tests/test_signal_diagram.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from diagram_types import Edge, ExtractionResult  # noqa: E402
import diagram_extract_ast  # noqa: E402
import diagram_extract_md  # noqa: E402


COMPILE_DB = REPO_ROOT / "build" / "compile_commands.json"
SIM_ROOT = REPO_ROOT / "sim"
TIMING_DOC = REPO_ROOT / "resources" / "timing_discipline.md"


class SignalDiagramTests(unittest.TestCase):
    """Pin a small set of architectural invariants the diagram should
    surface regardless of refactors. New modules / edges are fine; the
    listed ones must remain present with the listed classification."""

    @classmethod
    def setUpClass(cls) -> None:
        if not COMPILE_DB.exists():
            raise unittest.SkipTest(
                f"{COMPILE_DB} not found; run `cmake -B build && "
                f"cmake --build build` first"
            )
        cls.ast_result: ExtractionResult = diagram_extract_ast.extract(
            COMPILE_DB, SIM_ROOT,
        )
        cls.md_result: ExtractionResult = diagram_extract_md.extract(
            TIMING_DOC,
        )

    def test_module_count(self) -> None:
        # 5 PipelineStage subclasses + 6 leaf ExecutionUnit subclasses +
        # 7 standalone modules = 18.
        self.assertEqual(len(self.ast_result.modules), 18)

    def test_module_set_matches_markdown(self) -> None:
        ast_names = {m.name for m in self.ast_result.modules}
        md_names = {m.name for m in self.md_result.modules}
        self.assertEqual(ast_names, md_names)

    def test_known_edges_present_with_classification(self) -> None:
        """A floor of 10 high-signal edges that pin the diagram's spine.

        Each entry is (src, dst, classification). Under the two-axis
        model (Phase 0 of the naming-and-access-discipline refactor),
        the classification axis carries only REGISTERED or
        COMBINATIONAL; back-pressure direction is an architectural
        overlay derived by the renderer rather than a third
        classification. A failing assert here means either the
        architecture changed (update the test) or the AST extractor
        regressed (fix the extractor). Labels are intentionally not
        asserted — they're cosmetic."""
        ast_triples = {(e.src, e.dst, e.classification)
                       for e in self.ast_result.edges}
        expected = {
            ("FetchStage",        "BranchShadowTracker", "REGISTERED"),
            ("WarpScheduler",     "BranchShadowTracker", "REGISTERED"),
            ("BranchShadowTracker", "WarpScheduler",     "REGISTERED"),
            ("CoalescingUnit",    "L1Cache",             "REGISTERED"),
            ("L1Cache",           "CoalescingUnit",      "COMBINATIONAL"),
            ("WarpScheduler",     "Scoreboard",          "REGISTERED"),
            ("WritebackArbiter",  "Scoreboard",          "REGISTERED"),
            ("ALUUnit",           "WritebackArbiter",    "REGISTERED"),
            ("ALUUnit",           "PanicController",     "REGISTERED"),
            ("PanicController",   "WarpScheduler",       "REGISTERED"),
        }
        missing = expected - ast_triples
        self.assertFalse(
            missing,
            f"AST extractor missing expected edges: {sorted(missing)}",
        )

    def test_ast_and_markdown_are_equivalent(self) -> None:
        """The two extractors must produce the exact same set of
        (src, dst, classification) triples — no allow-list. Any future
        AST/markdown drift fails this assertion."""
        ast_edges = {(e.src, e.dst, e.classification)
                     for e in self.ast_result.edges}
        md_edges = {(e.src, e.dst, e.classification)
                    for e in self.md_result.edges}

        only_ast = ast_edges - md_edges
        only_md = md_edges - ast_edges

        self.assertFalse(
            only_ast,
            f"AST has edges not in markdown: {sorted(only_ast)}",
        )
        self.assertFalse(
            only_md,
            f"Markdown has edges not in AST: {sorted(only_md)}",
        )


if __name__ == "__main__":
    unittest.main()
