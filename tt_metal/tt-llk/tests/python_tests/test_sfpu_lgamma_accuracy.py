# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
fp32 lgamma accuracy across the domain, for tt-metal#55356.

The kernel evaluates Stirling's asymptotic series directly at the input with no
argument shift, so for z below ~0.75 (outside the Taylor bridge around z=1) it runs
Stirling far outside its useful range. The claim under test is a large error in the
middle of the domain, peaking near x = 0.5.

Drives calculate_lgamma_stirling_fp32, which returns the unadjusted Stirling value:
lgamma(z) with z = x for x >= 0.5, z = 1-x for x < 0.5. For x >= 0.5 that is also the
final ttnn output, since lgamma_adjusted_tile is the identity there.
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

FORMATS = input_output_formats([DataFormat.Float32], same=True)
ELEMENTS_PER_TILE = 1024

# The five spot values tabulated in the report.
_REPORTED = {
    0.3: (1.10061812, 1.09579799, 34433),
    0.4: (0.81208384, 0.79667782, 220913),
    0.6: (0.38282779, 0.39823386, 474555),
    0.7: (0.25604704, 0.26086725, 226137),
    1.5: (-0.12079416, -0.12078224, 1080),
}


def _reduced(x: float) -> float:
    """The z the kernel actually runs Stirling at.

    The 1-x must be done in float32, exactly as the kernel's `z = 1.0f - in` does, or
    the reference for x and its mirror 1-x lands on two different z and the symmetry
    check compares against two different exact values.
    """
    if x >= 0.5:
        return x
    return float(numpy.float32(1.0) - numpy.float32(x))


def _run(formats, xs: torch.Tensor) -> torch.Tensor:
    """calculate_lgamma_stirling_fp32 over one tile of inputs."""
    zs = torch.where(xs >= 0.5, xs, 1.0 - xs)
    log_z = torch.log(zs.to(torch.float64)).to(torch.float32)

    configuration = TestConfig(
        "sources/sfpu_lgamma_stirling_fp32_test.cpp",
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
            log_z,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=formats.input_format.is_32_bit(),
        compile_time_formats=True,
    )
    res = configuration.run().result[:ELEMENTS_PER_TILE]
    return torch.tensor(res, dtype=format_dict[formats.output_format]).flatten().to(torch.float32)


def _errors(device: torch.Tensor, xs: torch.Tensor):
    """(ulp, relative) against math.lgamma of the reduced argument, in float64."""
    ulps, rels = [], []
    for got, x in zip(device.tolist(), xs.tolist()):
        exact = math.lgamma(_reduced(float(x)))
        near = numpy.float32(exact)
        if not math.isfinite(near) or near == 0.0:
            ulps.append(0.0)
            rels.append(0.0)
            continue
        ulps.append(abs(got - exact) / abs(float(numpy.spacing(near, dtype=numpy.float32))))
        rels.append(abs(got - exact) / abs(exact))
    return ulps, rels


def _final_lgamma(stirling: float, x: float) -> float:
    """What lgamma_adjusted_tile makes of the Stirling value.

    For x >= 0.5 it is the identity. For x < 0.5 it applies the reflection,
    ln(pi) - log|sin(pi x)| - stirling. Done here in float64, so the device's own
    sin and log contribute nothing -- this is the best case for the kernel.
    """
    if x >= 0.5:
        return stirling
    return math.log(math.pi) - math.log(abs(math.sin(math.pi * x))) - stirling


def _tile(values) -> torch.Tensor:
    vals = list(values)
    reps = ELEMENTS_PER_TILE // len(vals) + 1
    return torch.tensor((vals * reps)[:ELEMENTS_PER_TILE], dtype=torch.float32)


@parametrize(formats=FORMATS, dummy=[0])
def test_lgamma_reported_spot_values(formats, dummy):
    """The five values tabulated on the issue, recomputed on this device."""
    wanted = list(_REPORTED)
    xs = _tile(wanted)
    device = _run(formats, xs)
    ulps, rels = _errors(device, xs)

    print("\n   x    returned      exact         rel err    ULP          reported: returned      exact          ULP")
    worst = 0.0
    for i, x in enumerate(wanted):
        xf = float(xs[i])
        final = _final_lgamma(device[i].item(), xf)
        exact = math.lgamma(xf)
        ulp = abs(final - exact) / abs(float(numpy.spacing(numpy.float32(exact), dtype=numpy.float32)))
        rel = abs(final - exact) / abs(exact)
        worst = max(worst, ulp)
        r_got, r_exact, r_ulp = _REPORTED[x]
        print(
            f"  {x:.1f}  {final: .8f}  {exact: .8f}  {rel:.2e}  {ulp:11,.0f}    "
            f"{r_got: .8f}  {r_exact: .8f}  {r_ulp:10,}"
        )

    assert worst <= 3.0, f"lgamma peaks at {worst:,.0f} ULP over the reported spot values"


@parametrize(
    formats=FORMATS,
    window=[
        ("mid", 0.05, 0.99),
        ("bridge-1", 0.75, 1.25),
        ("bridge-2", 1.75, 2.25),
        ("stirling-large", 2.5, 20.0),
    ],
)
def test_lgamma_domain_sweep(formats, window):
    """Dense sweep. The report says the error is confined to the middle of the domain."""
    name, lo, hi = window
    xs = torch.linspace(lo, hi, ELEMENTS_PER_TILE, dtype=torch.float32)
    device = _run(formats, xs)
    ulps, rels = _errors(device, xs)

    worst = max(ulps)
    at = xs[ulps.index(worst)].item()
    peak_rel = max(rels)
    at_rel = xs[rels.index(peak_rel)].item()
    over = sum(1 for u in ulps if u > 3.0)
    print(
        f"\n[{name}] x in [{lo}, {hi}]: max {worst:,.0f} ULP at x={at:.4f}, "
        f"peak {peak_rel:.2e} relative at x={at_rel:.4f}, {over}/{len(ulps)} over 3 ULP"
    )

    assert worst <= 3.0, f"{name} window: {worst:,.0f} ULP at x={at:.4f}"


@parametrize(formats=FORMATS, dummy=[0])
def test_lgamma_error_is_even_about_half(formats, dummy):
    """x and 1-x reduce to the same z, so an argument-shift defect makes the error even."""
    left = torch.linspace(0.05, 0.49, 512, dtype=torch.float32)
    xs = _tile(torch.cat([left, 1.0 - left]).tolist())
    device = _run(formats, xs)
    _, rels = _errors(device, xs)

    pairs = [(rels[i], rels[i + 512]) for i in range(8)]
    print("\n  x        1-x      rel err (x)   rel err (1-x)")
    for i, (a, b) in enumerate(pairs):
        print(f"  {left[i]:.4f}  {1.0 - left[i]:.4f}   {a:.4e}    {b:.4e}")

    mismatched = [i for i in range(512) if abs(rels[i] - rels[i + 512]) > 1e-12]
    assert not mismatched, f"{len(mismatched)}/512 mirror pairs differ"


@parametrize(formats=FORMATS, dummy=[0])
def test_lgamma_error_profile(formats, dummy):
    """Where the damage actually is, as final lgamma(x) vs float64, binned over x."""
    edges = [0.02, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.25, 1.75, 2.25, 4.0, 20.0]
    xs = torch.linspace(0.02, 20.0, ELEMENTS_PER_TILE, dtype=torch.float32)
    device = _run(formats, xs)

    print("\n  x range            max ULP      peak relative   which path")

    for lo, hi in zip(edges, edges[1:]):
        rows = [
            (abs(_final_lgamma(g, float(x)) - math.lgamma(float(x)))
             / abs(float(numpy.spacing(numpy.float32(math.lgamma(float(x))), dtype=numpy.float32))),
             abs(_final_lgamma(g, float(x)) - math.lgamma(float(x))) / abs(math.lgamma(float(x))))
            for g, x in zip(device.tolist(), xs.tolist())
            if lo <= float(x) < hi and abs(math.lgamma(float(x))) > 1e-6
        ]
        if not rows:
            continue
        z = _reduced((lo + hi) / 2)
        if abs(z - 1) <= 0.25:
            which = "Taylor bridge z~1"
        elif abs(z - 2) <= 0.25:
            which = "Taylor bridge z~2"
        else:
            which = f"Stirling at z~{z:.2f}"
        print(f"  [{lo:5.2f}, {hi:5.2f})   {max(r[0] for r in rows):11,.0f}   {max(r[1] for r in rows):.2e}      {which}")


@parametrize(formats=FORMATS, dummy=[0])
def test_lgamma_reported_peak(formats, dummy):
    """The peak the report quotes: 9.04e-2 relative at x = 0.5129."""
    probes = [0.5129, 0.5104, 0.5388, 0.5, 0.505, 0.52, 0.55, 0.6]
    xs = _tile(probes)
    device = _run(formats, xs)

    print("\n     x      final lgamma      exact        rel err       ULP")
    for i, x in enumerate(probes):
        xf = float(xs[i])
        final = _final_lgamma(device[i].item(), xf)
        exact = math.lgamma(xf)
        ulp = abs(final - exact) / abs(float(numpy.spacing(numpy.float32(exact), dtype=numpy.float32)))
        print(f"  {x:.4f}  {final: .8f}  {exact: .8f}   {abs(final - exact) / abs(exact):.4e}  {ulp:11,.0f}")
