# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
bf16 lgamma accuracy, the second half of tt-metal#55356's scope.

DataType::BFLOAT16 selects lgamma_fast_kernel.cpp, which drives calculate_lgamma_stirling
-- a different function from the fp32 path's calculate_lgamma_stirling_fp32. It has only
the r0/r1 Bernoulli terms and *no* polynomial bridge at all, so Stirling runs across the
whole reduced domain rather than just below z=0.75.

Error is reported in bfloat16 ULP (the output width) and as relative error, which is the
metric that compares across the two dtypes.
"""

import math

import numpy
import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, VectorMode, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import APPROX_MODE, VECTOR_MODE, generate_input_dim

pytestmark = [skip_for_quasar]

FORMATS = input_output_formats([DataFormat.Float16_b], same=True)
ELEMENTS_PER_TILE = 1024


def _reduced(x: float) -> float:
    """The z the kernel runs Stirling at; the 1-x is done in float32 as the kernel does."""
    if x >= 0.5:
        return x
    return float(numpy.float32(1.0) - numpy.float32(x))


def _final_lgamma(stirling: float, x: float) -> float:
    """lgamma_adjusted_tile's job, done here in float64 (best case for the kernel)."""
    if x >= 0.5:
        return stirling
    return math.log(math.pi) - math.log(abs(math.sin(math.pi * x))) - stirling


def _bf16_ulp(value: float) -> float:
    """Width of one bfloat16 ULP at `value`."""
    v = torch.tensor(value, dtype=torch.bfloat16)
    nxt = torch.nextafter(v, torch.tensor(float("inf"), dtype=torch.bfloat16))
    return abs(float(nxt) - float(v))


def _run(formats, xs: torch.Tensor, dest_acc) -> torch.Tensor:
    configuration = TestConfig(
        "sources/sfpu_lgamma_stirling_bf16_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            APPROX_MODE(ApproximationMode.No),
            VECTOR_MODE(VectorMode.RC),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            xs,
            formats.input_format,
            torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32),
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
        compile_time_formats=True,
    )
    res = configuration.run().result[:ELEMENTS_PER_TILE]
    return torch.tensor(res, dtype=format_dict[formats.output_format]).flatten().to(torch.float32)


def _errors(device, xs):
    """(bf16 ULP, relative) of the final lgamma against float64 on the bf16-rounded input."""
    ulps, rels = [], []
    for got, x in zip(device.tolist(), xs.tolist()):
        final = _final_lgamma(got, float(x))
        exact = math.lgamma(float(x))
        if abs(exact) < 1e-6 or not math.isfinite(final):
            continue
        width = _bf16_ulp(exact)
        ulps.append(abs(final - exact) / width if width else 0.0)
        rels.append(abs(final - exact) / abs(exact))
    return ulps, rels


DEST_ACC = [DestAccumulation.No, DestAccumulation.Yes]

# lgamma has roots at x = 1 and x = 2. Relative error and ULP-of-result both diverge
# there for arithmetic reasons rather than accuracy ones, so those metrics are only
# asserted where the function is comfortably away from zero.
_ROOT_GUARD = 0.05


@parametrize(formats=FORMATS, dest_acc=DEST_ACC)
def test_lgamma_bf16_error_profile(formats, dest_acc):
    """Where the bf16 path's error lives, binned over x."""
    edges = [0.02, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.25, 1.75, 2.25, 4.0, 20.0]
    xs = torch.linspace(0.02, 20.0, ELEMENTS_PER_TILE, dtype=torch.bfloat16).to(torch.float32)
    device = _run(formats, xs, dest_acc)

    print(f"\n  dest_acc={dest_acc.name}")
    print("  x range          max bf16 ULP   peak relative   (near a root)")
    worst_ulp = 0.0
    for lo, hi in zip(edges, edges[1:]):
        sel = [(g, x) for g, x in zip(device.tolist(), xs.tolist()) if lo <= float(x) < hi]
        if not sel:
            continue
        u, r = _errors(torch.tensor([g for g, _ in sel]), torch.tensor([x for _, x in sel]))
        if not u:
            continue
        near_root = any(abs(math.lgamma(float(x))) < _ROOT_GUARD for _, x in sel)
        if not near_root:
            worst_ulp = max(worst_ulp, max(u))
        print(f"  [{lo:5.2f}, {hi:5.2f})     {max(u):9,.1f}     {max(r):.2e}     {'yes' if near_root else ''}")

    assert worst_ulp <= 4.0, f"bf16 lgamma peaks at {worst_ulp:.1f} ULP away from its roots"


@parametrize(formats=FORMATS, dest_acc=DEST_ACC)
def test_lgamma_bf16_spot_values(formats, dest_acc):
    """The fp32 report's spot values, on the bf16 path."""
    probes = [0.3, 0.4, 0.5129, 0.6, 0.7, 1.5, 3.0, 8.0]
    vals = [float(torch.tensor(p, dtype=torch.bfloat16)) for p in probes]
    reps = ELEMENTS_PER_TILE // len(vals) + 1
    xs = torch.tensor((vals * reps)[:ELEMENTS_PER_TILE], dtype=torch.float32)
    device = _run(formats, xs, dest_acc)

    print(f"\n  dest_acc={dest_acc.name}")
    print("     x       returned      exact        rel err     bf16 ULP")
    worst = 0.0
    for i, x in enumerate(vals):
        final = _final_lgamma(device[i].item(), x)
        exact = math.lgamma(x)
        rel = abs(final - exact) / abs(exact)
        ulp = abs(final - exact) / _bf16_ulp(exact)
        if abs(exact) >= _ROOT_GUARD:
            worst = max(worst, ulp)
        print(f"  {x:7.4f}  {final: .6f}  {exact: .6f}   {rel:.3e}  {ulp:9,.1f}")

    assert worst <= 4.0, f"bf16 lgamma peaks at {worst:.1f} ULP over the spot values"


@parametrize(formats=FORMATS, dummy=[0])
def test_lgamma_bf16_absolute_error(formats, dummy):
    """Absolute error, which is the meaningful metric near lgamma's roots at x=1 and x=2.

    Relative error and ULP-of-result both blow up there for arithmetic reasons, not
    accuracy ones, so the profile above overstates those two bands.
    """
    xs = torch.linspace(0.02, 20.0, ELEMENTS_PER_TILE, dtype=torch.bfloat16).to(torch.float32)
    device = _run(formats, xs, DestAccumulation.No)

    rows = []
    for got, x in zip(device.tolist(), xs.tolist()):
        xf = float(x)
        final = _final_lgamma(got, xf)
        exact = math.lgamma(xf)
        if not math.isfinite(final):
            continue
        # bf16 quantisation of the output alone costs this much; anything at or below
        # it is the format, not the kernel.
        floor = _bf16_ulp(exact) / 2 if exact != 0.0 else 0.0
        rows.append((xf, abs(final - exact), floor))

    edges = [0.02, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.25, 1.75, 2.25, 4.0, 20.0]
    print("\n  x range          max |abs err|   bf16 repr floor   excess over format")
    worst = 0.0
    for lo, hi in zip(edges, edges[1:]):
        sel = [r for r in rows if lo <= r[0] < hi]
        if not sel:
            continue
        err = max(r[1] for r in sel)
        at = max(sel, key=lambda r: r[1])
        worst = max(worst, err)
        ratio = err / at[2] if at[2] else float("inf")
        print(f"  [{lo:5.2f}, {hi:5.2f})     {err:.3e}       {at[2]:.3e}        {ratio:8.1f}x")

    print(f"\n  worst absolute error anywhere on [0.02, 20]: {worst:.3e}")


@parametrize(formats=FORMATS, dummy=[0])
def test_lgamma_bf16_reported_bands(formats, dummy):
    """The follow-up report's bf16 bands: 'up to 20% on [0.5,1) and 76% on [1,2)'.

    Reported as relative error, absolute error, and distance from the nearest root.
    lgamma has roots at x = 1 and x = 2; relative error is unbounded beside them for
    arithmetic reasons, so the question is whether the peaks sit on the roots.
    """
    for lo, hi in ((0.5, 1.0), (1.0, 2.0)):
        xs = torch.linspace(lo, hi, ELEMENTS_PER_TILE, dtype=torch.bfloat16).to(torch.float32)
        device = _run(formats, xs, DestAccumulation.No)
        rows = []
        for got, x in zip(device.tolist(), xs.tolist()):
            xf = float(x)
            if xf <= 0:
                continue
            exact = math.lgamma(xf)
            if exact == 0.0:
                continue
            final = _final_lgamma(got, xf)
            rows.append((abs(final - exact) / abs(exact), abs(final - exact), xf, exact))
        rel, absol, at, ex = max(rows)
        worst_abs = max(r[1] for r in rows)
        root_dist = min(abs(at - 1.0), abs(at - 2.0))
        # the same band, excluding a neighbourhood of the roots
        away = [r for r in rows if min(abs(r[2] - 1.0), abs(r[2] - 2.0)) > 0.05]
        print(
            f"\n  [{lo}, {hi}) peak relative {rel * 100:.1f}% at x={at:.4f} "
            f"(|lgamma| there = {abs(ex):.3e}, distance to nearest root = {root_dist:.4f})"
            f"\n      absolute error at that point: {absol:.3e};  max absolute over band: {worst_abs:.3e}"
            f"\n      peak relative excluding |x-root| <= 0.05: {max(r[0] for r in away) * 100:.2f}%"
        )
