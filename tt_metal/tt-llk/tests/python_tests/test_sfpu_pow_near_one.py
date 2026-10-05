# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Near-1.0 base accuracy for both fp32 power kernels.

Both compute base**pow as 2**(pow * log2(base)) and reduce the base mantissa m to
z = (m - 1)/(m + 1). The question under test is how z is formed:

    z = m * recip - recip      one SFPMAD, cancels ~24 bits for m near 1
    z = (m - 1.0f) * recip     separate subtract, exact by Sterbenz on m in [0.5, 2]

Driven through the production entry points:
    binary  calculate_sfpu_binary_pow   (power_binary_tile)
    unary   calculate_unary_power       (power_tile)

Reference is IEEE pow in float64 on the exact float32 inputs; the error is reported in
float32 ULP of that reference.
"""

import math
import os
import struct

import numpy

import pytest
import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    VectorMode,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    SFPU_UNARY_SCALAR,
    VECTOR_MODE,
    generate_input_dim,
)

pytestmark = [skip_for_quasar]

# FP32 in and out; DestAccumulation.Yes is the only valid mode for a 32-bit input.
FORMATS = input_output_formats([DataFormat.Float32], same=True)

ELEMENTS_PER_TILE = 1024

# (exponent, half-width of the k sweep). The half-widths keep 2**(pow*log2(base))
# inside the fp32 range: at pow = 3e6 a base 512 ULP from 1.0 would need 2**264.
# (exponent, half-width of the k sweep, ULP bound).
#
# The bounds are the measured Wormhole-silicon behaviour of the separate subtract with
# headroom, not a target: the residual at the largest exponents is the pow*log2(base)
# rounding the kernel's two-sum path does not yet cover, and it is unrelated to the
# range-reduction cancellation this test guards. The MAD form exceeds every one of them
# by three to four orders of magnitude.
#
# The half-widths keep 2**(pow*log2(base)) inside the fp32 range: at pow = 3e6 a base
# 512 ULP from 1.0 would need 2**264.
_CASES = [
    # Narrow: 64 bases, the sweep the bounty write-up reports.
    (1.0e3, 32, 3.0),
    (1.0e5, 32, 3.0),
    (1.0e6, 32, 16.0),
    (3.0e6, 32, 48.0),
    # Wide: a full tile of distinct bases.
    (1.0e3, 512, 3.0),
    (1.0e5, 512, 32.0),
    (1.0e6, 512, 200.0),
    (3.0e6, 64, 48.0),
]

# Documented bound from PR #49649 for non-integer exponents; reported, not asserted,
# because the largest exponents here sit outside it for an unrelated reason.
_PR_49649_BOUND = 3.0


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _near_one_bases(half_width: int) -> torch.Tensor:
    """One tile of bases 1 + k*2**-23, k cycling over [-half_width, half_width]."""
    step = 2.0**-23
    ks = list(range(-half_width, half_width + 1))
    values = [float(torch.tensor(1.0 + k * step, dtype=torch.float32)) for k in ks]
    reps = ELEMENTS_PER_TILE // len(values) + 1
    tile = (values * reps)[:ELEMENTS_PER_TILE]
    return torch.tensor(tile, dtype=torch.float32)


def _ulp_error(device: torch.Tensor, bases: torch.Tensor, exponent: float):
    """float32 ULP distance from the float64 reference, per element, plus relative error."""
    errors = []
    relative = []
    for got, base in zip(device.tolist(), bases.tolist()):
        reference = math.pow(float(base), float(exponent))  # float64
        nearest = numpy.float32(reference)
        if not math.isfinite(nearest) or not math.isfinite(got):
            # Overflow on either side is outside what this sweep measures.
            continue
        # float32 ULP of the reference, not float64: numpy.spacing keeps the width.
        ulp = float(numpy.spacing(nearest, dtype=numpy.float32))
        errors.append(abs(got - reference) / ulp)
        relative.append(abs(got - reference) / abs(reference))
    return errors, relative


def _run_binary(formats, bases: torch.Tensor, exponent: float) -> torch.Tensor:
    exponents = torch.full((ELEMENTS_PER_TILE,), exponent, dtype=torch.float32)
    configuration = TestConfig(
        "sources/sfpu_pow_near_one_binary_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            APPROX_MODE(ApproximationMode.No),
            VECTOR_MODE(VectorMode.RC),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            bases,
            formats.input_format,
            exponents,
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
    res_from_L1 = configuration.run().result[:ELEMENTS_PER_TILE]
    torch_format = format_dict[formats.output_format]
    return torch.tensor(res_from_L1, dtype=torch_format).flatten().to(torch.float32)


def _run_unary(formats, bases: torch.Tensor, exponent: float) -> torch.Tensor:
    configuration = TestConfig(
        "sources/sfpu_pow_near_one_unary_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            APPROX_MODE(ApproximationMode.No),
            SFPU_UNARY_SCALAR(_bits(exponent)),
            VECTOR_MODE(VectorMode.RC),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            bases,
            formats.input_format,
            torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32),
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
    res_from_L1 = configuration.run().result[:ELEMENTS_PER_TILE]
    torch_format = format_dict[formats.output_format]
    return torch.tensor(res_from_L1, dtype=torch_format).flatten().to(torch.float32)


_RUNNERS = {"binary": _run_binary, "unary": _run_unary}


@parametrize(
    formats=FORMATS,
    path=list(_RUNNERS),
    case=_CASES,
)
def test_sfpu_pow_near_one(formats, path, case):
    """base**pow for bases within a few hundred ULP of 1.0 and a large exponent."""
    exponent, half_width, ulp_bound = case
    bases = _near_one_bases(half_width)
    device = _RUNNERS[path](formats, bases, exponent)

    errors, relative = _ulp_error(device, bases, exponent)
    assert errors, "every lane overflowed; the sweep measured nothing"

    worst = max(errors)
    over = sum(1 for e in errors if e > _PR_49649_BOUND)
    mean = sum(errors) / len(errors)
    print(
        f"\n[{path}] pow={exponent:g} bases=1+k*2**-23 |k|<={half_width}: "
        f"max {worst:.1f} ULP ({max(relative):.2e} relative), mean {mean:.2f} ULP, "
        f"{over}/{len(errors)} over {_PR_49649_BOUND} ULP"
    )

    assert worst <= ulp_bound, (
        f"{path} path: {worst:.1f} ULP at pow={exponent:g} (bound {ulp_bound}), "
        f"{over}/{len(errors)} lanes over the {_PR_49649_BOUND} ULP bound of PR #49649"
    )


# =============================================================================
# Ordinary domain: the separate subtract must not cost anything away from 1.0.
# =============================================================================

_ORDINARY_BASES = 48
_ORDINARY_EXPONENTS = 16


def _ordinary_stimuli():
    """Bases log-spaced over [0.01, 100] crossed with exponents over [-15, 15]."""
    bases = torch.logspace(-2, 2, _ORDINARY_BASES, dtype=torch.float32)
    exponents = torch.linspace(-15, 15, _ORDINARY_EXPONENTS, dtype=torch.float32)
    grid = torch.cartesian_prod(bases, exponents)
    reps = ELEMENTS_PER_TILE // grid.shape[0] + 1
    grid = grid.repeat(reps, 1)[:ELEMENTS_PER_TILE]
    return grid[:, 0].contiguous(), grid[:, 1].contiguous()


def _run_binary_pairs(formats, bases, exponents):
    configuration = TestConfig(
        "sources/sfpu_pow_near_one_binary_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            APPROX_MODE(ApproximationMode.No),
            VECTOR_MODE(VectorMode.RC),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            bases,
            formats.input_format,
            exponents,
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
    res_from_L1 = configuration.run().result[:ELEMENTS_PER_TILE]
    torch_format = format_dict[formats.output_format]
    return torch.tensor(res_from_L1, dtype=torch_format).flatten().to(torch.float32)


@parametrize(formats=FORMATS, dummy=[0])
def test_sfpu_pow_ordinary_domain(formats, dummy):
    """base**pow away from 1.0, as a guard against the fix costing accuracy elsewhere."""
    bases, exponents = _ordinary_stimuli()
    device = _run_binary_pairs(formats, bases, exponents)

    errors = []
    for got, base, exponent in zip(device.tolist(), bases.tolist(), exponents.tolist()):
        reference = math.pow(float(base), float(exponent))
        nearest = numpy.float32(reference)
        if not math.isfinite(nearest) or not math.isfinite(got) or nearest == 0.0:
            continue
        ulp = float(numpy.spacing(nearest, dtype=numpy.float32))
        errors.append(abs(got - reference) / ulp)

    dump = os.environ.get("POW_DUMP")
    if dump:
        numpy.save(dump, device.numpy())

    worst = max(errors)
    mean = sum(errors) / len(errors)
    print(
        f"\n[binary] ordinary domain, bases [0.01, 100] x exponents [-15, 15], "
        f"{len(errors)} finite lanes: max {worst:.1f} ULP, mean {mean:.2f} ULP"
    )
    assert worst <= 64.0, f"ordinary-domain pow regressed to {worst:.1f} ULP"
