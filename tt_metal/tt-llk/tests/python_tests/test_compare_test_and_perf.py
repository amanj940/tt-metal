# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pairing and ignore-set checks for compare_test_and_perf.

These do not import SFPU test modules. The comparer itself loads those, and
that import needs a device stack this host does not have.
"""

from pathlib import Path

import compare_test_and_perf as cmp


def test_normalize_strips_prefix_and_quasar_suffix():
    assert (
        cmp.normalize("test_perf_eltwise_binary_sfpu_float_quasar")
        == "eltwise_binary_sfpu_float"
    )
    assert cmp.normalize("test_perf_eltwise_binary_sfpu_float") == (
        "eltwise_binary_sfpu_float"
    )
    assert cmp.normalize("test_eltwise_binary_sfpu_float") == (
        "eltwise_binary_sfpu_float"
    )


def test_ignore_set_includes_implied_math_format_separately_from_measurement():
    assert "implied_math_format" in cmp.IGNORED_AXES
    assert "implied_math_format" not in cmp.MEASUREMENT_AXES
    for axis in ("iterations", "loop_factor", "run_types", "is_perf"):
        assert axis in cmp.IGNORED_AXES
        assert axis in cmp.MEASUREMENT_AXES
    assert cmp.ignored_reason("loop_factor") == "ignored measurement axis"
    assert cmp.ignored_reason("implied_math_format") == "ignored"


def test_cross_arch_pairs_quasar_suffix(tmp_path: Path):
    (tmp_path / "perf_matmul.py").write_text("x = 1\n")
    (tmp_path / "perf_only_bh.py").write_text("x = 1\n")
    quasar = tmp_path / "quasar"
    quasar.mkdir()
    (quasar / "perf_matmul_quasar.py").write_text("x = 1\n")
    (quasar / "perf_only_qsr_quasar.py").write_text("x = 1\n")

    matched, left_only, right_only = cmp.discover_cross_arch(
        tmp_path, "blackhole", "quasar", "perf"
    )

    assert [(key, left.name, right.name) for key, left, right in matched] == [
        ("matmul", "perf_matmul.py", "perf_matmul_quasar.py")
    ]
    assert [path.name for path in left_only] == ["perf_only_bh.py"]
    assert [path.name for path in right_only] == ["perf_only_qsr_quasar.py"]


def test_cross_arch_same_tree_pairs_a_file_with_itself(tmp_path: Path):
    (tmp_path / "test_matmul.py").write_text("x = 1\n")
    (tmp_path / "perf_matmul.py").write_text("x = 1\n")

    matched, left_only, right_only = cmp.discover_cross_arch(
        tmp_path, "wormhole", "blackhole", "func"
    )

    assert len(matched) == 1
    assert matched[0][0] == "matmul"
    assert matched[0][1] == matched[0][2]
    assert left_only == []
    assert right_only == []


def test_cross_arch_verdict_uses_architecture_names():
    bucket, headline = cmp.verdict(
        "mathop",
        ["add", "mul"],
        ["add"],
        True,
        True,
        "axis",
        cmp.arch_sides("blackhole", "quasar"),
    )
    assert bucket == "diff"
    assert headline.startswith("[~] mathop: quasar subset of blackhole")

    _bucket, only = cmp.verdict(
        "bcast_dim",
        ["none"],
        [],
        True,
        False,
        "axis",
        cmp.arch_sides("blackhole", "quasar"),
    )
    assert only == "[blackhole] bcast_dim: blackhole-only axis"


def test_same_arch_verdict_keeps_functional_perf_labels():
    _bucket, headline = cmp.verdict(
        "approx_mode", [], ["No"], False, True, "axis", cmp.FUNCTIONAL_PERF
    )
    assert headline == "[P] approx_mode: PERF-ONLY axis"
