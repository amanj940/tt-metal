// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <ranges>
#include <vector>

namespace tt::tt_metal::streaming_profiler {

// The least-squares line y = y_mean + slope * (x - x_mean).
struct LineFit {
    double x_mean = 0.0, y_mean = 0.0, slope = 0.0;
    constexpr double at(double x) const { return y_mean + slope * (x - x_mean); }
};
template <
    std::ranges::forward_range Points,
    std::invocable<std::ranges::range_reference_t<const Points>> X,
    std::invocable<std::ranges::range_reference_t<const Points>> Y>
LineFit fit_line(const Points& points, X x, Y y) {
    long double sum_x = 0;  // long double because x may be a count that is large compared with its spread
    double sum_y = 0.0;
    size_t count = 0;
    for (const auto& point : points) {
        sum_x += std::invoke(x, point);
        sum_y += std::invoke(y, point);
        count++;
    }
    LineFit fit{
        .x_mean = static_cast<double>(sum_x / static_cast<long double>(count)),
        .y_mean = sum_y / static_cast<double>(count)};
    double sum_xx = 0.0, sum_xy = 0.0;
    for (const auto& point : points) {
        const double dx = std::invoke(x, point) - fit.x_mean;
        sum_xx += dx * dx;
        sum_xy += dx * (std::invoke(y, point) - fit.y_mean);
    }
    fit.slope = sum_xy / sum_xx;
    return fit;
}

// The least-squares solution x or, if there isn't one, the first unknown with no chain of edges to the ground.
struct Potential {
    std::vector<double> x;
    std::optional<size_t> unreached;
};
// Each edge says x[plus(edge)] - x[minus(edge)] = value(edge). An empty end is the ground, which is fixed at 0.
// The least-squares normal equations are the graph's Laplacian with the ground's row and column removed, solved by
// Cholesky factoring them into their lower triangle in place.
template <
    std::ranges::forward_range Edges,
    std::invocable<std::ranges::range_reference_t<const Edges>> Plus,
    std::invocable<std::ranges::range_reference_t<const Edges>> Minus,
    std::invocable<std::ranges::range_reference_t<const Edges>> Value>
Potential solve_potential(const Edges& edges, Plus plus_of, Minus minus_of, Value value_of, size_t unknowns) {
    // A pivot this small means its unknown has no chain of edges to the ground.
    constexpr double kSingularPivot = 1e-9;
    std::vector<double> normal(unknowns * unknowns, 0.0);
    const auto entry = [&](size_t row, size_t col) -> double& { return normal[row * unknowns + col]; };
    Potential potential{.x = std::vector<double>(unknowns, 0.0)};
    std::vector<double>& x = potential.x;
    for (const auto& edge : edges) {
        const std::optional<size_t> plus = std::invoke(plus_of, edge);
        const std::optional<size_t> minus = std::invoke(minus_of, edge);
        const double value = std::invoke(value_of, edge);
        if (plus) {
            entry(*plus, *plus) += 1.0;
            x[*plus] += value;
        }
        if (minus) {
            entry(*minus, *minus) += 1.0;
            x[*minus] -= value;
        }
        if (plus && minus) {
            entry(*plus, *minus) -= 1.0;
            entry(*minus, *plus) -= 1.0;
        }
    }
    for (size_t j = 0; j < unknowns; j++) {
        double pivot = entry(j, j);
        for (size_t k = 0; k < j; k++) {
            pivot -= entry(j, k) * entry(j, k);
        }
        if (pivot <= kSingularPivot) {
            potential.unreached = j;
            return potential;
        }
        entry(j, j) = std::sqrt(pivot);
        for (size_t i = j + 1; i < unknowns; i++) {
            double below = entry(i, j);
            for (size_t k = 0; k < j; k++) {
                below -= entry(i, k) * entry(j, k);
            }
            entry(i, j) = below / entry(j, j);
        }
    }
    for (size_t i = 0; i < unknowns; i++) {
        for (size_t k = 0; k < i; k++) {
            x[i] -= entry(i, k) * x[k];
        }
        x[i] /= entry(i, i);
    }
    for (size_t i = unknowns; i-- > 0;) {
        for (size_t k = i + 1; k < unknowns; k++) {
            x[i] -= entry(k, i) * x[k];
        }
        x[i] /= entry(i, i);
    }
    return potential;
}

}  // namespace tt::tt_metal::streaming_profiler
