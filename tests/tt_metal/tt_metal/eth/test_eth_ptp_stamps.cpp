// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tests the Blackhole Ethernet 1588 layer on every active eth link between two chips: 256 round trips with one-step
// egress stamps and an RX stamp rule, then 16 two-step stamped frames. It needs slow dispatch
// (TT_METAL_SLOW_DISPATCH_MODE=1) on a multi-chip Blackhole system.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>

#include "device_fixture.hpp"
#include "eth_test_common.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_ptp_stamps.hpp"

namespace tt::tt_metal {
namespace {

using eth_ptp_stamps::Result;

constexpr const char* kKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_ptp_stamps.cpp";
constexpr double kStampTickNs = 20.0;

// One round trip's four stamps. Each end's stamps are on its own PTP time.
struct RoundTrip {
    double transmitter_egress, receiver_ingress, receiver_egress, transmitter_ingress;
};

std::vector<RoundTrip> round_trips(const Result& transmitter, const Result& receiver) {
    std::vector<RoundTrip> trips;
    for (uint32_t i = 0; i < eth_ptp_stamps::kRounds; i++) {
        trips.push_back(
            {.transmitter_egress = static_cast<double>(receiver.stamps[i].peer_egress),
             .receiver_ingress = static_cast<double>(receiver.stamps[i].ingress),
             .receiver_egress = static_cast<double>(transmitter.stamps[i].peer_egress),
             .transmitter_ingress = static_cast<double>(transmitter.stamps[i].ingress)});
    }
    return trips;
}

// Compares only stamps from the same end.
void expect_causal(std::span<const RoundTrip> trips, const std::string& link) {
    for (size_t i = 0; i < trips.size(); i++) {
        const RoundTrip& trip = trips[i];
        EXPECT_LT(trip.transmitter_egress, trip.transmitter_ingress) << link << " round " << i;
        EXPECT_LT(trip.receiver_ingress, trip.receiver_egress) << link << " round " << i;
        if (i > 0) {
            EXPECT_GT(trip.transmitter_egress, trips[i - 1].transmitter_ingress) << link << " round " << i;
            EXPECT_GT(trip.receiver_ingress, trips[i - 1].receiver_egress) << link << " round " << i;
        }
    }
}

// Each stamp is its true time rounded down to the 20 ns grid, so a correctly paired round trip is within a tick of the
// truth and never two ticks from the median one-way time or from the line fitted to the offsets. A missing or
// mispaired stamp is off by far more.
void expect_steady(std::span<const RoundTrip> trips, const std::string& link) {
    const auto n = static_cast<double>(trips.size());
    std::vector<double> one_way, at, offset;
    double at_mean = 0, offset_mean = 0;
    for (const RoundTrip& trip : trips) {
        one_way.push_back(
            0.5 *
            ((trip.transmitter_ingress - trip.transmitter_egress) - (trip.receiver_egress - trip.receiver_ingress)));
        at.push_back(0.5 * (trip.transmitter_egress + trip.transmitter_ingress));
        offset.push_back(
            0.5 *
            ((trip.receiver_ingress + trip.receiver_egress) - (trip.transmitter_egress + trip.transmitter_ingress)));
        at_mean += at.back() / n;
        offset_mean += offset.back() / n;
    }
    std::vector<double> sorted = one_way;
    std::ranges::sort(sorted);
    const double one_way_median = sorted[sorted.size() / 2];
    double at_sq = 0, at_offset = 0;
    for (size_t i = 0; i < trips.size(); i++) {
        at_sq += (at[i] - at_mean) * (at[i] - at_mean);
        at_offset += (at[i] - at_mean) * (offset[i] - offset_mean);
    }
    const double slope = at_offset / at_sq;
    for (size_t i = 0; i < trips.size(); i++) {
        EXPECT_LT(std::abs(one_way[i] - one_way_median), 2 * kStampTickNs) << link << " round " << i;
        EXPECT_LT(std::abs(offset[i] - (offset_mean + slope * (at[i] - at_mean))), 2 * kStampTickNs)
            << link << " round " << i;
    }
}

void run_link(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_a,
    const std::shared_ptr<distributed::MeshDevice>& mesh_b,
    const CoreCoord& eth_a,
    const CoreCoord& eth_b) {
    IDevice* dev_a = mesh_a->get_devices()[0];
    IDevice* dev_b = mesh_b->get_devices()[0];
    const uint32_t base =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t result_addr = base + eth_ptp_stamps::kResultOffset;
    std::vector<uint32_t> zero(sizeof(Result) / sizeof(uint32_t), 0);
    detail::WriteToDeviceL1(dev_a, eth_a, result_addr, zero, CoreType::ETH);
    detail::WriteToDeviceL1(dev_b, eth_b, result_addr, zero, CoreType::ETH);

    const auto make_workload = [&](const CoreCoord& core, bool transmitter) {
        Program program;
        EthernetConfig config{.compile_args = {transmitter ? 1u : 0u}};
        eth_test_common::set_arch_specific_eth_config(config);
        const KernelHandle kernel = CreateKernel(program, kKernel, core, config);
        SetRuntimeArgs(program, kernel, core, {base});
        distributed::MeshWorkload workload;
        const distributed::MeshCoordinate zero_coord(0, 0);
        workload.add_program(distributed::MeshCoordinateRange(zero_coord, zero_coord), std::move(program));
        return workload;
    };
    distributed::MeshWorkload transmitter_workload = make_workload(eth_a, true);
    distributed::MeshWorkload receiver_workload = make_workload(eth_b, false);
    std::thread transmitter_thread([&] { fixture->RunProgram(mesh_a, transmitter_workload); });
    std::thread receiver_thread([&] { fixture->RunProgram(mesh_b, receiver_workload); });
    transmitter_thread.join();
    receiver_thread.join();

    const std::string link =
        fmt::format("chip {} eth {} -> chip {} eth {}", dev_a->id(), eth_a.str(), dev_b->id(), eth_b.str());
    const auto read = [&](IDevice* dev, const CoreCoord& core) {
        Result result{};
        detail::ReadFromDeviceL1(
            dev, core, result_addr, std::span(reinterpret_cast<uint8_t*>(&result), sizeof(result)), CoreType::ETH);
        return result;
    };
    const Result transmitter = read(dev_a, eth_a);
    const Result receiver = read(dev_b, eth_b);

    for (const auto& [result, end] : {std::pair{&transmitter, "transmitter"}, std::pair{&receiver, "receiver"}}) {
        ASSERT_EQ(result->done, eth_ptp_stamps::kDone) << link << ": the " << end << " kernel did not finish";
        EXPECT_EQ(result->header_select_after, result->header_select_before)
            << link << ": the " << end << "'s TX header row selection was not restored";
        EXPECT_EQ(result->no_match_after, result->no_match_before)
            << link << ": the " << end << "'s RX no-match actions were not restored";
        // Both clocks are read just after a refclk update, so they agree to the ns when the offset is right.
        EXPECT_EQ(result->restart_error_ns, 0) << link << ": the " << end << "'s PTP time is "
                                               << result->restart_error_ns << " ns off what the restart's offset says";
        ASSERT_EQ(result->rounds, eth_ptp_stamps::kRounds)
            << link << ": the " << end << " stopped waiting for its peer";
        EXPECT_EQ(result->unstamped, 0u) << link << ": frames to the " << end << " without an egress stamp";
        EXPECT_EQ(result->ingress_mismatched, 0u)
            << link << ": " << end << " frames whose ingress stamp wasn't the RX FIFO's one entry";
    }
    EXPECT_EQ(transmitter.two_step_good, eth_ptp_stamps::kTwoStepFrames)
        << link << ": two-step frames without a matching, in-window TX FIFO entry";

    const std::vector<RoundTrip> trips = round_trips(transmitter, receiver);
    expect_causal(trips, link);
    expect_steady(trips, link);
}

}  // namespace

TEST_F(MeshDeviceFixture, ActiveEthPtpStamps) {
    if (arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "1588 stamping is Blackhole's";
    }
    auto& cluster = MetalContext::instance().get_cluster();
    std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> mesh_of;
    for (const auto& mesh : devices_) {
        mesh_of.emplace(mesh->get_device_ids()[0], mesh);
    }
    size_t links = 0;
    for (const auto& [chip_a, mesh_a] : mesh_of) {
        for (const auto& [chip_b, cores] : cluster.get_ethernet_cores_grouped_by_connected_chips(chip_a)) {
            const auto mesh_b = mesh_of.find(chip_b);
            if (chip_b <= chip_a || mesh_b == mesh_of.end()) {
                continue;
            }
            for (const CoreCoord& eth_a : cores) {
                const CoreCoord eth_b =
                    std::get<1>(cluster.get_connected_ethernet_core(std::make_tuple(chip_a, eth_a)));
                run_link(this, mesh_a, mesh_b->second, eth_a, eth_b);
                links++;
            }
        }
    }
    if (links == 0) {
        GTEST_SKIP() << "no ethernet link between local chips";
    }
}

}  // namespace tt::tt_metal
