# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the ULP regression check in ``helpers/ulp_budget_diff.py``.

No device and no torch: that module runs on a slim CI runner against two revisions of
a text file. Pinned here: which changes count as a regression, and that its parse of
the live table agrees with the real loader's -- it reads the file a second way, so the
two are free to drift unless something says they may not.
"""

import ast
import sys
from datetime import date
from pathlib import Path

import pytest
from helpers.ulp_budget_diff import (
    _MAX_ROWS,
    DuplicateRow,
    _measured_cells,
    _nonfinite_cells,
    compare,
    parse_table,
    recorded_max,
    recorded_nonfinite,
    render_budget_diff,
    render_headroom,
)
from helpers.ulp_provenance import Provenance

_HEADER = "Abs:  # a header is prose: no code reads it"
_BASE_ROWS = (
    "{in: Float16_b, out: Float16_b, max_ulp: 1, measured: 1, run: 2026-01-01}",
    "{in: Float16, out: Float16_b, max_ulp: 4, measured: 3, run: 2026-01-01}",
    "{in: Bfp8_b, out: Bfp8_b, metric: tolerance, measured: 393, run: 2026-01-01}",
    "{in: Float32, out: Float32, max_ulp: 2, measured: 1, run: 2026-01-01}",
)
_BF16 = (("in", "Float16_b"), ("out", "Float16_b"))


def _head(*rows):
    return _HEADER + "\n" + "".join(f"  - {r}\n" for r in rows)


_BASE = _head(*_BASE_ROWS)


def _edited(index, row):
    """The base table with one row replaced."""
    rows = list(_BASE_ROWS)
    rows[index] = row
    return _head(*rows)


def _changes(base, head):
    return compare(parse_table(base), parse_table(head))


def _kinds(base, head):
    return {c.cell[1]: c.kind for c in _changes(base, head)}


def _measured(*cells):
    """``(in, out, max)`` triples as the recorder's rows, folded to one worst per cell."""
    return _measured_cells(
        [
            {"op": "Abs", "in": i, "out": o, "approx": None, "dest": None, "max": m}
            for i, o, m in cells
        ]
    )


def _refuses(kind):
    """The suite's ``expect_error`` fixture needs a device; these are host-only tests."""
    return pytest.raises(kind)  # allow-pytest.raises: host-only test


# ── What counts as a regression ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "provenance, remeasured",
    [
        ("measured: 1, run: 2026-01-01", False),
        ("measured: 1, run: 2026-01-01}  # re-measured 2026-02-02, honestly", False),
        ("measured: 8, run: 2026-02-02", True),
        ("measured: 1, run: 2026-02-02", True),
    ],
    ids=["untouched", "only-the-prose", "re-measured", "same-figure-new-run"],
)
def test_a_raised_budget_is_a_regression_marked_by_its_provenance(
    provenance, remeasured
):
    """The raise is always surfaced. Whether the row's measurement fields changed with
    it is what separates a re-measurement from a number fitted to a failure; editing
    the prose beside the row does not count."""
    if "}" in provenance:
        body, _, comment = provenance.partition("}")
        row = f"{{in: Float16_b, out: Float16_b, max_ulp: 9, {body}}}{comment}"
    else:
        row = f"{{in: Float16_b, out: Float16_b, max_ulp: 9, {provenance}}}"
    head = _edited(0, row)
    (change,) = _changes(_BASE, head)
    assert change.kind == "raised" and change.is_regression
    assert change.remeasured is remeasured


def test_losing_the_gate_entirely_is_a_regression():
    """Demotion to tolerance is the loosest change: the cell stops being judged on a
    step budget at all, and a bounds check cannot see it."""
    head = _edited(
        0,
        "{in: Float16_b, out: Float16_b, metric: tolerance, measured: 14337, run: 2026-01-01}",
    )
    assert _kinds(_BASE, head)[_BF16] == "ungated"


def test_deleting_a_gated_row_is_a_regression_and_a_tolerance_row_is_not():
    """A row that gated nothing cannot be a loss when it goes."""
    kinds = _kinds(_BASE, _head(_BASE_ROWS[1], _BASE_ROWS[3]))
    assert kinds[_BF16] == "removed"
    assert (("in", "Bfp8_b"), ("out", "Bfp8_b")) not in kinds


def test_tightening_and_newly_gating_are_not_regressions():
    head = _head(
        "{in: Float16_b, out: Float16_b, max_ulp: 0, measured: 0, run: 2026-01-01}",
        _BASE_ROWS[1],
        "{in: Bfp8_b, out: Bfp8_b, max_ulp: 3, measured: 2, run: 2026-01-01}",
        _BASE_ROWS[3],
        "{in: Float16, out: Float32, max_ulp: 7, measured: 6, run: 2026-01-01}",
    )
    changes = _changes(_BASE, head)
    assert not any(c.is_regression for c in changes)
    assert {c.kind for c in changes} == {"tightened", "gated"}


def test_a_row_the_sweep_collapses_is_not_a_loss_while_its_cells_keep_their_budget():
    """The emitter drops a key dimension when both values measure alike, so a re-emit
    deletes `approx`-keyed rows and writes one without. Nothing any query resolves to
    has changed, and a row diff reported 16 of these as regressions."""
    base = _head(
        '{in: Float16_b, out: Float32, approx: "No", dest: "Yes", max_ulp: 36018, measured: 32744, run: 2026-01-01}  # a',
        '{in: Float16_b, out: Float32, approx: "Yes", dest: "Yes", max_ulp: 36018, measured: 32744, run: 2026-01-01}  # b',
    )
    same = _head(
        '{in: Float16_b, out: Float32, dest: "Yes", max_ulp: 36018, measured: 32744, run: 2026-01-01}  # c'
    )
    changes = _changes(base, same)
    assert not any(c.is_regression for c in changes)
    # The one honest difference: a query with `approx` unset matched neither keyed row
    # before and matches the collapsed one now, so that variant is newly gated.
    assert [(c.kind, c.variants) for c in changes] == [("gated", 1)]
    # A raise through the collapsed row is one change covering both variants,
    # re-measured because the deciding row's measurement is new.
    raised = _head(
        '{in: Float16_b, out: Float32, dest: "Yes", max_ulp: 36030, measured: 32755, '
        "run: 2026-02-02}"
    )
    (change,) = [c for c in _changes(base, raised) if c.is_regression]
    assert change.kind == "raised" and change.variants == 2 and change.remeasured


def test_a_more_specific_row_that_loosens_a_cell_is_a_raise_not_an_addition():
    """The hole a row diff cannot see: no existing row changes, yet every Float32 query
    now resolves to 1000 steps instead of 2. Demoting the same cells through a new
    tolerance row is the same hole, and the loosest gate of all."""
    base = _head("{max_ulp: 2}  # exact")
    loosened = _head("{max_ulp: 2}  # exact", "{in: Float32, max_ulp: 1000}  # wide")
    (change,) = _changes(base, loosened)
    assert change.kind == "raised" and change.cell == ("Abs", (("in", "Float32"),))
    demoted = _head("{max_ulp: 2}  # exact", "{in: Float32, metric: tolerance}  # off")
    assert [c.kind for c in _changes(base, demoted)] == ["ungated"]


_FLOOR_CASES = [
    (
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 1.0e-07, measured: 1, run: 2026-01-01}",
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 5.0e-05, measured: 1, run: 2026-01-01}",
        "floor_widened",
    ),
    (
        "{in: Float16_b, out: Float16_b, max_ulp: 2, measured: 1, run: 2026-01-01}",
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 5.0e-05, measured: 1, run: 2026-01-01}",
        "floor_widened",
    ),
    # A smaller max_ulp must not mask a wider floor: the floor can more than pay for it.
    (
        "{in: Float16_b, out: Float16_b, max_ulp: 8, near_zero_atol: 1.0e-07, measured: 1, run: 2026-01-01}",
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 5.0e-02, measured: 1, run: 2026-01-01}",
        "floor_widened",
    ),
    (
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 5.0e-05, measured: 1, run: 2026-01-01}",
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 1.0e-07, measured: 1, run: 2026-01-01}",
        None,
    ),
    # On a tolerance row nothing consults the floor.
    (
        "{in: Float16_b, out: Float16_b, metric: tolerance, measured: 393, run: 2026-01-01}",
        "{in: Float16_b, out: Float16_b, metric: tolerance, near_zero_atol: 0.5, measured: 393, run: 2026-01-01}",
        None,
    ),
]


@pytest.mark.parametrize(
    "base_row, head_row, kind",
    _FLOOR_CASES,
    ids=[
        "widened",
        "introduced",
        "wider-floor-tighter-budget",
        "narrowed",
        "tolerance-row",
    ],
)
def test_the_near_zero_floor_is_part_of_the_gate(base_row, head_row, kind):
    """`ulp_elementwise_valid` accepts a lane inside `near_zero_atol` however many steps
    out it is, so widening the floor loosens the gate and narrowing it does not. A guard
    that watched only `max_ulp` would call every one of these tables unchanged."""
    changes = _changes(_head(base_row), _head(head_row))
    assert [c.kind for c in changes] == ([kind] if kind else [])
    assert all(c.is_regression for c in changes)


def test_the_report_shows_the_floor_on_both_sides():
    """Or both columns read "2" and a widened-floor row looks inert."""
    base_row, head_row, _ = _FLOOR_CASES[0]
    text = render_budget_diff(
        _changes(_head(base_row), _head(head_row)), "ulp-budget-raise-approved"
    )
    assert "1e-07" in text and "5e-05" in text


def test_an_unchanged_table_reports_nothing():
    assert _changes(_BASE, _BASE) == []
    assert "No budget changed" in render_budget_diff([], "the-label")


def test_the_report_names_the_cell_the_change_and_the_label():
    head = _head(
        "{in: Float16_b, out: Float16_b, max_ulp: 9, measured: 1, run: 2026-01-01}"
    )
    report = render_budget_diff(_changes(_BASE, head), "ulp-budget-raise-approved")
    assert "loosen a gate" in report
    assert "in: Float16_b, out: Float16_b" in report
    assert "budget raised" in report
    assert "ulp-budget-raise-approved" in report
    assert "**no**" in report  # not re-measured


# ── Reading the table ─────────────────────────────────────────────────────────


def test_an_anchor_and_its_alias_are_both_real_rows():
    """The coarse-LUT tolerance pair is a YAML anchor and an alias, and a line-oriented
    reader skipped both -- silently, being tolerance rows with no budget to compare. A
    budget hidden behind an alias would have been invisible; the loader tie-back below
    is what caught it."""
    table = parse_table(
        "GeluAppx:\n"
        "  - &shared {max_ulp: 3, measured: 2, run: 2026-01-01}\n"
        "SigmoidAppx:\n"
        "  - *shared  # same cause, same number\n"
    )
    assert {cell[0] for cell in table} == {"GeluAppx", "SigmoidAppx"}
    assert all(row.max_ulp == 3 for row in table.values())
    # The alias is the anchor's row, measurement included: re-measuring one is
    # re-measuring both.
    shared = Provenance(measured=2, run=date(2026, 1, 1))
    assert {row.provenance for row in table.values()} == {shared}


def test_a_merge_key_row_is_read_through():
    """The registry loads the table with PyYAML's SafeLoader, which resolves `<<`."""
    table = parse_table(
        "Base:\n"
        "  - &b {out: Float16_b, max_ulp: 2, measured: 1, run: 2026-01-01}\n"
        "Abs:\n"
        "  - {<<: *b, max_ulp: 5, measured: 4, run: 2026-01-01}\n"
    )
    assert table[("Abs", (("out", "Float16_b"),))].max_ulp == 5


def test_only_a_rows_own_fields_mark_a_re_measurement():
    """A row's measurement is its own fields; a changed header line or comment beside
    it is prose and re-measures nothing, so it cannot launder a raise."""

    def table(header, budget, provenance):
        return parse_table(
            f"Abs:  # {header}\n"
            f"  - {{out: Float32, max_ulp: {budget}, {provenance}}}  # prose\n"
        )

    base = table("sweep A", 1, "measured: 1, run: 2026-01-01")
    for head, remeasured in (
        (table("sweep B", 4, "measured: 1, run: 2026-01-01"), False),
        (table("sweep A", 4, "measured: 3, run: 2026-02-02"), True),
    ):
        (raised,) = [c for c in compare(base, head) if c.is_regression]
        assert raised.kind == "raised" and raised.remeasured is remeasured


def test_an_unquoted_yaml_boolean_names_the_same_cell_as_a_quoted_one():
    """``dest: Yes`` is a YAML boolean; the registry maps it to the enum member
    (test_a_quoted_and_an_unquoted_no_mean_the_same_thing). Keyed as "True" here, a
    raise written that way would be reported against a cell that does not exist."""
    quoted = parse_table(
        'Abs:\n  - {in: Float16_b, out: Float16_b, dest: "Yes", max_ulp: 9}  # c\n'
    )
    bare = parse_table(
        "Abs:\n  - {in: Float16_b, out: Float16_b, dest: Yes, max_ulp: 9}  # c\n"
    )
    assert set(quoted) == set(bare)
    assert dict(next(iter(bare))[1])["dest"] == "Yes"


def test_a_not_measurable_row_still_records_the_measurable_lanes_maximum():
    """The emitter writes both numbers as fields; each reader takes its own, and the
    finite lanes of a demoted cell stay judged."""
    table = parse_table(
        'I1:\n  - {in: Float16_b, out: Float32, dest: "No", metric: tolerance, '
        "measured: 7, nonfinite: 31120, run: 2026-01-01}  # not measurable: lanes "
        "disagreeing with the golden about being finite (x=91: inf -> -1.16e37)\n"
    )
    row = next(iter(table.values()))
    assert recorded_nonfinite(row) == 31120
    assert recorded_max(row) == 7


def test_a_duplicated_cell_is_refused_rather_than_judged():
    """`_load_table` rejects two rows of equal specificity, so the table cannot load.
    Keeping the last row silently produced a verdict -- a *tightening*, even -- for a
    table the registry would not accept at all."""
    with _refuses(DuplicateRow) as caught:
        parse_table(
            _head(
                "{in: Float16_b, out: Float16_b, max_ulp: 9, measured: 1, run: 2026-01-01}",
                "{in: Float16_b, out: Float16_b, max_ulp: 1, measured: 1, run: 2026-01-01}",
            )
        )
    assert caught.value.cell == ("Abs", _BF16)


# ── The headroom half ─────────────────────────────────────────────────────────


def test_several_measurements_of_one_cell_keep_the_worst():
    """A driver enumerates axes the budget key does not, so one cell is recorded more
    than once. Keeping the last would hide the worst of them."""
    cells = _measured(
        ("Float16_b", "Float16_b", 3),
        ("Float16_b", "Float16_b", 11),
        ("Float16_b", "Float16_b", 5),
    )
    assert list(cells.values()) == [11]


@pytest.mark.parametrize(
    "budget, measured, expect",
    [(1, 9, "over budget"), (1, 1, "no headroom"), (8, 0, "could tighten")],
    ids=["over", "tight", "slack"],
)
def test_the_headroom_report_classifies_against_the_declared_budget(
    budget, measured, expect
):
    table = parse_table(
        f"Abs:\n  - {{in: Float16_b, out: Float16_b, max_ulp: {budget}}}\n"
    )
    report, over = render_headroom(
        table, _measured(("Float16_b", "Float16_b", measured))
    )
    assert expect in report
    assert over == (1 if expect == "over budget" else 0)


def test_an_exact_cell_at_its_zero_budget_is_not_a_warning():
    """`0 == 0` is an op exact by construction doing what it claims, enrolled so any
    drift fails. Calling that "no headroom" buried the real ones on a 130-cell run."""
    table = parse_table("Abs:\n  - {in: Float16_b, out: Float16_b, max_ulp: 0}\n")
    report, over = render_headroom(table, _measured(("Float16_b", "Float16_b", 0)))
    assert over == 0
    assert "no headroom" not in report
    assert "headroom to spare" in report


def test_a_tolerance_cell_is_judged_against_its_recorded_measurement():
    """No budget, so the sweep passes it whatever it measures -- this is the only place
    a regression there can show. The row's own ``measured`` is the baseline; a cell
    whose row records none is not judged at all."""
    table = parse_table(
        "Abs:\n"
        "  - {in: Float16_b, out: Float16_b, metric: tolerance, measured: 393, "
        "run: 2026-01-01}\n"
        "  - {in: Float16, out: Float16_b, metric: tolerance}  # max 9 ULP, prose\n"
    )
    report, over = render_headroom(table, _measured(("Float16_b", "Float16_b", 393)))
    assert over == 0 and "regressed" not in report
    report, over = render_headroom(
        table,
        _measured(("Float16_b", "Float16_b", 500), ("Float16", "Float16_b", 10**6)),
    )
    assert over == 1  # the un-baselined cell is not counted
    assert "| 500 | 393 | regressed |" in report
    assert "Regressed on the tolerance metric" in report


def test_a_measurement_resolves_against_the_most_specific_row():
    """Same rule the registry uses, so the report judges a cell against the budget that
    gates it rather than a broader one."""
    table = parse_table(
        "Abs:\n"
        "  - {out: Float16_b, max_ulp: 100}  # broad\n"
        "  - {in: Float16_b, out: Float16_b, max_ulp: 1}  # specific\n"
    )
    _, over = render_headroom(table, _measured(("Float16_b", "Float16_b", 50)))
    assert over == 1  # 50 is inside the broad 100 but past the specific 1


def test_a_long_report_is_capped_and_says_how_many_it_withheld():
    """A PR comment has a size limit, and 2,000-odd gated cells could blow past it."""
    n = _MAX_ROWS + 5
    table = "Abs:\n" + "".join(
        f'  - {{in: Float16_b, out: Float16_b, dest: "{i}", max_ulp: 8}}\n'
        for i in range(n)
    )
    rows = [
        {"op": "Abs", "in": "Float16_b", "out": "Float16_b", "dest": str(i), "max": 99}
        for i in range(n)
    ]
    report, over = render_headroom(parse_table(table), _measured_cells(rows))
    assert over == n
    assert "5 more" in report
    assert report.count("over budget |") == _MAX_ROWS


# ── The slim runner, and the tie-back to the real loader ──────────────────────


#: The modules the slim runner loads, by path: the tool, and the one sibling it reads
#: the table through.
_SLIM_MODULES = ("ulp_budget_diff.py", "ulp_provenance.py")


@pytest.mark.parametrize("module", _SLIM_MODULES)
def test_the_tool_imports_nothing_but_the_standard_library_and_yaml(module):
    """The PR check runs on a slim runner: no torch, no ttexalens, no LLK venv. Read off
    each module's own AST: the import list is the actual property, and a subprocess
    would prove it for one environment only. The one sibling allowed is
    ``ulp_provenance``, itself held to the same rule."""
    tool = Path(__file__).parent / "helpers" / module
    roots = set()
    for node in ast.walk(ast.parse(tool.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            # A relative import would reach back into `helpers`; only the sibling the
            # runner also has on its path is allowed.
            if node.level:
                assert (node.level, node.module) == (1, "ulp_provenance"), node.module
                continue
            if node.module:
                roots.add(node.module.split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"yaml", "ulp_provenance"}
    assert roots <= allowed, f"imports outside stdlib + yaml: {sorted(roots - allowed)}"
    assert "helpers" not in roots and "torch" not in roots


def test_the_tool_runs_by_path_with_its_sibling(tmp_path):
    """What the workflow does: ``python3 helpers/ulp_budget_diff.py``, so the package
    is not importable and the sibling is found on the script's own directory."""
    import subprocess

    tool = Path(__file__).parent / "helpers" / "ulp_budget_diff.py"
    (tmp_path / "base.yaml").write_text(_BASE, encoding="utf-8")
    (tmp_path / "head.yaml").write_text(_edited(0, _BASE_ROWS[0]), encoding="utf-8")
    done = subprocess.run(
        [
            sys.executable,
            str(tool),
            "diff",
            "--base",
            str(tmp_path / "base.yaml"),
            "--head",
            str(tmp_path / "head.yaml"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


def test_the_workflow_invokes_the_tool_by_path_not_as_a_module():
    """`helpers/__init__.py` imports ttexalens, which the slim runner does not have, so
    `-m helpers.ulp_budget_diff` would fail there however clean this module's own
    imports are."""
    repo = Path(__file__).resolve().parents[4]
    workflow = repo / ".github/workflows/llk-sfpu-ulp-budget-guard.yaml"
    if not workflow.exists():  # pragma: no cover - the guard ships with the workflow
        pytest.skip("workflow not in this checkout")
    # Comments stripped: the workflow explains in prose why it does *not* use `-m`.
    body = "\n".join(
        line
        for line in workflow.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "helpers/ulp_budget_diff.py diff" in body
    assert "-m helpers.ulp_budget_diff" not in body
    # A change to the parser changes the guard's verdicts, so it re-runs the guard.
    for module in _SLIM_MODULES:
        assert f"tt_metal/tt-llk/tests/python_tests/helpers/{module}" in body


def test_this_parse_agrees_with_the_registry_loader_on_the_live_table():
    """This module reads the table a second way -- text, no torch -- so the CI check can
    run without the LLK environment and against an arbitrary base revision. Nothing
    stops the two parses drifting except this. Compared as (op, key, budget), which is
    exactly what a regression is defined over."""
    from helpers.sfpu_accuracy_budget import _SFPU_ACCURACY_BUDGET, _TABLE_PATH, Metric

    mine = {
        (cell[0], cell[1], row.max_ulp)
        for cell, row in parse_table(_TABLE_PATH.read_text(encoding="utf-8")).items()
    }
    fields = (
        ("in", "input_format"),
        ("out", "output_format"),
        ("approx", "approx_mode"),
        ("dest", "dest_acc"),
        ("arch", "arch"),
    )
    theirs = set()
    for op, table in _SFPU_ACCURACY_BUDGET.items():
        for key, contract in table.items():
            pinned = tuple(
                (short, getattr(key, attr).name)
                for short, attr in fields
                if getattr(key, attr) is not None
            )
            budget = contract.max_ulp if contract.metric == Metric.ULP else None
            theirs.add((op.name, pinned, budget))
    assert mine == theirs, (
        f"only this parser sees: {sorted(mine - theirs)[:5]}\n"
        f"only the loader sees: {sorted(theirs - mine)[:5]}"
    )


def test_this_resolution_agrees_with_the_registry_on_every_swept_cell():
    """Agreeing on the rows is not agreeing on what a query resolves to, and a
    regression is defined over the latter. Every cell the sweep drives, for every
    enrolled unary op, must land on the same row through ``_resolve`` as through the
    registry's ``_winner``."""
    from helpers.sfpu_accuracy_budget import (
        _SFPU_ACCURACY_BUDGET,
        _TABLE_PATH,
        MEASURED_ARCH,
        BudgetKey,
        _winner,
    )
    from helpers.ulp_budget_diff import _resolve
    from helpers.ulp_sweep import sweep_cells

    table = parse_table(_TABLE_PATH.read_text(encoding="utf-8"))
    compared = 0
    for op, rows in _SFPU_ACCURACY_BUDGET.items():
        for in_fmt, out_fmt, approx, dest in sweep_cells():
            query = BudgetKey(
                approx_mode=approx,
                input_format=in_fmt,
                output_format=out_fmt,
                dest_acc=dest,
                arch=MEASURED_ARCH,
            )
            found = _winner(rows, query, op.name)
            theirs = None
            if found is not None:
                key = found[0]
                theirs = tuple(
                    (short, getattr(key, attr).name)
                    for short, attr in (
                        ("in", "input_format"),
                        ("out", "output_format"),
                        ("approx", "approx_mode"),
                        ("dest", "dest_acc"),
                        ("arch", "arch"),
                    )
                    if getattr(key, attr) is not None
                )
            asked = (
                ("in", in_fmt.name),
                ("out", out_fmt.name),
                ("approx", approx.name),
                ("dest", dest.name),
                ("arch", MEASURED_ARCH.name),
            )
            mine = _resolve(table, op.name, asked)
            assert (mine.key if mine else None) == theirs, (op.name, asked)
            compared += 1
    from helpers.ulp_sweep import sweep_cells as _cells

    assert compared == len(_cells()) * len(_SFPU_ACCURACY_BUDGET)


def test_the_headroom_report_fails_an_overflow_the_row_does_not_account_for():
    """A "not measurable" row records how many lanes went non-finite; the sweep skips
    such a tolerance cell but records the count, and more lanes than the row names --
    or any on a row that names none -- is a regression the nightly must fail on."""
    table = parse_table(
        "Exp:\n"
        '  - {in: Float32, out: Bfp8_b, dest: "Yes", metric: tolerance, measured: 0, '
        "nonfinite: 2, run: 2026-01-01}  # not measurable: x=3e38: 3e38 -> inf\n"
        '  - {in: Float16, out: Float16, dest: "No", metric: tolerance, measured: 511, '
        "run: 2026-01-01}\n"
        "Exp2:\n"
        '  - {in: Float32, out: Bfp8_b, dest: "Yes", metric: tolerance, nonfinite: 2}'
        "  # a parked row without a figure for the rest\n"
    )

    def judged(cells=None):
        measurements = [
            {"op": "Exp", "in": i, "out": o, "dest": d, "max": 0, "nonfinite": n}
            for (i, o, d), n in (cells or {}).items()
        ]
        _, regressions = render_headroom(
            table, _measured_cells(measurements), _nonfinite_cells(measurements)
        )
        return regressions

    at_row = ("Float32", "Bfp8_b", "Yes")
    plain = ("Float16", "Float16", "No")
    assert judged() == 0
    assert judged({at_row: 2}) == 0  # what the row accounts for
    assert judged({at_row: 3}) == 1  # one more lane than it names
    assert judged({plain: 1}) == 1  # any on a row that names none
    # The count stands on its own, with or without a figure for the rest of the cell.
    (exp2,) = [row for (op, _), row in table.items() if op == "Exp2"]
    assert (recorded_nonfinite(exp2), recorded_max(exp2)) == (2, None)
    # A measurement written before the count existed judges as before.
    _, regressions = render_headroom(
        table,
        _measured_cells(
            [{"op": "Exp", "in": "Float16", "out": "Float16", "dest": "No", "max": 3}]
        ),
    )
    assert regressions == 0
