# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Wide-domain fp32 lgamma sweep, for the #55356 before/after comparison.

Drives calculate_lgamma_stirling_fp32 directly (see test_sfpu_lgamma_accuracy.py for
why that is the production fp32 path). Prints a band table rather than asserting, so
the same file can be run against the shipped kernel and a candidate fix.
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


def _run(formats, xs: torch.Tensor) -> torch.Tensor:
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
            xs, formats.input_format, log_z, formats.input_format, formats.output_format,
            tile_count_A=1, tile_count_B=1, tile_count_res=1,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=formats.input_format.is_32_bit(),
        compile_time_formats=True,
    )
    res = configuration.run().result[:ELEMENTS_PER_TILE]
    return torch.tensor(res, dtype=format_dict[formats.output_format]).flatten().to(torch.float32)


def _band(formats, lo, hi, log_spaced=False):
    """max ULP / peak relative over one band of z (>= 0.5, so no reflection needed)."""
    if log_spaced:
        xs = torch.from_numpy(numpy.geomspace(lo, hi, ELEMENTS_PER_TILE, dtype=numpy.float32))
    else:
        xs = torch.linspace(lo, hi, ELEMENTS_PER_TILE, dtype=torch.float32)
    got = _run(formats, xs)
    worst_u = worst_r = 0.0
    at_u = at_r = float("nan")
    for g, x in zip(got.tolist(), xs.tolist()):
        if not math.isfinite(x) or x <= 0:
            continue
        exact = math.lgamma(x)
        if abs(exact) < 1e-6:          # on a root, ULP-of-result is meaningless
            continue
        den = abs(float(numpy.spacing(numpy.float32(exact), dtype=numpy.float32)))
        u = abs(g - exact) / den
        r = abs(g - exact) / abs(exact)
        if u > worst_u:
            worst_u, at_u = u, x
        if r > worst_r:
            worst_r, at_r = r, x
    return worst_u, at_u, worst_r, at_r


BANDS = [
    (0.5, 0.75, False), (0.75, 1.0, False), (1.0, 1.25, False), (1.25, 1.75, False),
    (1.75, 2.25, False), (2.25, 3.0, False), (3.0, 3.5, False), (3.5, 4.0, False),
    (4.0, 10.0, False), (10.0, 100.0, True), (100.0, 1e4, True), (1e4, 1e10, True),
    (1e10, 1e30, True),
]


@parametrize(formats=FORMATS, dummy=[0])
def test_lgamma_wide_domain_band_table(formats, dummy):
    print("\n       z band            max ULP        at z        peak rel      at z")
    overall = 0.0
    for lo, hi, logsp in BANDS:
        u, au, r, ar = _band(formats, lo, hi, logsp)
        overall = max(overall, u)
        print(f"  [{lo:>9.4g}, {hi:<9.4g})  {u:11,.1f}  {au:10.4g}    {r:.2e}  {ar:10.4g}")
    print(f"\n  worst over all bands: {overall:,.1f} ULP")


@parametrize(formats=FORMATS, dummy=[0])
def test_lgamma_half_clamp_flat_band(formats, dummy):
    """The shipped kernel pins |z-0.5| < 0.01 to the constant lgamma(0.5).

    If the clamp is present the outputs over z in [0.5, 0.51) are all one value.
    """
    xs = torch.linspace(0.5, 0.5099, ELEMENTS_PER_TILE, dtype=torch.float32)
    got = _run(formats, xs)
    distinct = len(set(got.tolist()))
    spread = float(got.max() - got.min())
    exact_spread = math.lgamma(0.5) - math.lgamma(0.5099)
    print(
        f"\n  z in [0.5, 0.5099]: {distinct} distinct outputs over {ELEMENTS_PER_TILE} inputs, "
        f"output spread {spread:.3e}, true spread {exact_spread:.3e}"
    )
    worst = max(
        abs(g - math.lgamma(x)) / abs(float(numpy.spacing(numpy.float32(math.lgamma(x)), dtype=numpy.float32)))
        for g, x in zip(got.tolist(), xs.tolist())
    )
    print(f"  max {worst:,.0f} ULP inside the clamp window")
    assert distinct > 1, "output is a single constant across the window: the clamp is live"
