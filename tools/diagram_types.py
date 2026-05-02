"""Shared dataclasses for signal-flow diagram extractors and renderer.

Both `diagram_extract_md.py` and `diagram_extract_ast.py` populate these
types; `render_signal_diagram.py` consumes them.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Module:
    """A node in the signal-flow diagram.

    `name` is the canonical PascalCase class name. `cluster` is the
    presentation-layer grouping label.
    """
    name: str
    cluster: str


@dataclass
class Edge:
    """A directed edge between two modules.

    `classification` is one of REGISTERED, COMBINATIONAL, READY/STALL, or
    UNKNOWN. `source_row` is the markdown inventory row number when the
    edge originated from `diagram_extract_md.py`; the AST extractor leaves
    it `None`.
    """
    src: str
    dst: str
    classification: str
    label: str = ""
    source_row: int | None = None


@dataclass
class ExtractionResult:
    """Aggregated extractor output."""
    modules: list[Module] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
