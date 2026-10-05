// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"
#include "tt_metal/impl/streaming_profiler/kernels/eth_clock.hpp"

constexpr uint32_t kCtrlAddr = get_named_compile_time_arg_val("ctrl");
constexpr uint32_t kSyncRingAddr = get_named_compile_time_arg_val("sync_ring");

constexpr uint32_t kHistory = 8;         // updates held back so the ones leading up to a change can be sent too
constexpr uint32_t kDenseAfter = 64;     // updates sent around a change, including the kHistory before it
constexpr int64_t kOffLineEighths = 16;  // two ticks; a steady update is within a few eighths of the line
constexpr uint32_t kStopPollMask = 255u;

// The pad range covers a refclk tick (20 ns, which is 16 to 27 cycles across the AICLK range) as well as a block. With
// a smaller range, the loop's own length phase-locks the passes to the refclk, and each chip's updates settle on one
// AICLK cycle of the crossing.
constexpr uint32_t kRulerPadRange = 64;
constexpr int32_t kDropGap = -128;

// Calibration and sampling call the same 64-byte-aligned copy of the pass, so calibration measures exactly the reads
// that sample, regardless of the code around them.
__attribute__((noinline, aligned(64))) bool pass(uint32_t (&walls)[3], uint32_t (&refclks)[4]) {
    struct Block {
        uint32_t wall, first_refclk, second_refclk, last_refclk;
    };
    Block blocks[3];
    const auto read = [](Block& block) __attribute__((always_inline)) {
        block.first_refclk = eth_ptp::kRefclkLo.read();
        block.second_refclk = eth_ptp::kRefclkLo.read();
        block.wall = eth_ptp::kWallClockLo.read();
        block.last_refclk = eth_ptp::kRefclkLo.read();
    };
    bool stepped = false;
    eth_clock::disable_branch_predictor();
    read(blocks[2]);
    read(blocks[0]);
#pragma GCC unroll 20
    for (uint32_t k = 0; k < 20; k++) {
        Block& cur = blocks[(k + 1) % 3];
        const Block& prev = blocks[k % 3];
        const Block& prev2 = blocks[(k + 2) % 3];
        read(cur);
        if (prev.last_refclk != prev2.last_refclk) {
            walls[0] = prev2.wall;
            walls[1] = prev.wall;
            walls[2] = cur.wall;
            refclks[0] = prev2.last_refclk;
            refclks[1] = prev.first_refclk;
            refclks[2] = prev.second_refclk;
            refclks[3] = prev.last_refclk;
            stepped = true;
            break;
        }
    }
    eth_clock::enable_branch_predictor();
    return stepped;
}
// Runs one pass and returns 1 if it wrote an update to slot. With period 0, for calibration, the update written is the
// length of the block after the step, which is never the block right after the pad. Otherwise an update is written only
// if its blocks are period apart and its gap's position isn't kDropGap. It holds the new refclk and the step's wall
// time in eighths, which is the block's wall read plus the gap's position.
FORCE_INLINE uint32_t
run(uint32_t period, eth_clock::GapPositions gap_positions, volatile tt_l1_ptr uint32_t* slot, uint32_t& random_state) {
    uint32_t walls[3], refclks[4];
    eth_clock::nops<kRulerPadRange>(eth_clock::draw(random_state, kRulerPadRange));
    if (!pass(walls, refclks)) {
        return 0;
    }
    if (period == 0) {
        slot[0] = walls[2] - walls[1];
        return 1;
    }
    const uint32_t gap = refclks[1] != refclks[0] ? 0u : refclks[2] != refclks[1] ? 1u : 2u;
    const int32_t gap_position = gap_positions.at(gap);
    if (walls[1] - walls[0] != period || walls[2] - walls[1] != period || gap_position == kDropGap) {
        return 0;
    }
    slot[0] = refclks[gap + 1];
    slot[1] = (walls[1] << 3) + static_cast<uint32_t>(gap_position);
    return 1;
}

using PointWriter = eth_clock::ClockPointWriter<kCtrlAddr, kSyncRingAddr>;
constexpr kernel_profiler::SyncMeta kMetaDense{.dense = 1, .kind = kernel_profiler::SyncKind::Ruler};
constexpr kernel_profiler::SyncMeta kMetaThin{.kind = kernel_profiler::SyncKind::Ruler};

// Decides which updates to send. Placement error peaks within a few microseconds of a clock change, so every update
// around a change is sent (kDenseAfter of them). Elsewhere only one in kSyncRulerKeepEvery is sent, and the host
// weights each by that number. A change is an update more than kOffLineEighths off the rate measured at the last
// change. That rate comes from the pair of updates straddling the change, so the update after a real change triggers
// again, and the rate then settles on a clean pair.
struct Ruler {
    struct Update {
        uint64_t refclk, wall_eighths;
    };
    struct Output {
        PointWriter writer;
        uint32_t thin = 0, dense_left = 0;
        int64_t wall_per_refclk_eighths = 0;
        __attribute__((noinline)) void release(const Update& update) {
            if (dense_left != 0) {
                writer.add(
                    update.refclk, update.wall_eighths, 0, static_cast<uint32_t>(wall_per_refclk_eighths), kMetaDense);
            } else if (++thin == kernel_profiler::kSyncRulerKeepEvery) {
                thin = 0;
                writer.add(
                    update.refclk, update.wall_eighths, 0, static_cast<uint32_t>(wall_per_refclk_eighths), kMetaThin);
            }
        }
    };
    // References rather than members: release() takes the output's address and the history is indexed by a variable,
    // and either one held here would force the whole Ruler, counters included, into memory instead of registers.
    Output& output;
    Update (&held)[kHistory];
    uint32_t held_count = 0, held_next = 0;

    FORCE_INLINE void on_update(uint64_t refclk, uint64_t wall_eighths) {
        if (held_count != 0) {
            const Update& prev = held[(held_next + kHistory - 1) % kHistory];
            const auto refclk_step = static_cast<int64_t>(refclk - prev.refclk),
                       wall_eighths_step = static_cast<int64_t>(wall_eighths - prev.wall_eighths);
            const int64_t off_line_eighths = wall_eighths_step - output.wall_per_refclk_eighths * refclk_step;
            if (output.wall_per_refclk_eighths == 0 || off_line_eighths > kOffLineEighths ||
                off_line_eighths < -kOffLineEighths) {
                if (output.wall_per_refclk_eighths != 0) {
                    output.dense_left = kDenseAfter;
                    release_held();
                }
                output.wall_per_refclk_eighths = (wall_eighths_step + refclk_step / 2) / refclk_step;
            }
        }
        if (held_count == kHistory) {
            output.release(held[held_next]);
            held_count--;
        }
        held[held_next] = {refclk, wall_eighths};
        held_next = (held_next + 1) % kHistory;
        held_count++;
        if (output.dense_left != 0) {
            output.dense_left--;
        }
    }
    FORCE_INLINE void release_held() {
        for (; held_count != 0; held_count--) {
            output.release(held[(held_next + kHistory - held_count) % kHistory]);
        }
    }
};

void kernel_main() {
    volatile tt_l1_ptr kernel_profiler::ResidentCtrl* ctrl =
        reinterpret_cast<volatile tt_l1_ptr kernel_profiler::ResidentCtrl*>(kCtrlAddr);
    uint32_t random_state = eth_ptp::kWallClockLo.read() | 1u;
    eth_clock::Calibration calibration = eth_clock::calibrate(
        [&](uint32_t period, eth_clock::GapPositions gap_positions, volatile tt_l1_ptr uint32_t* slot) {
            return run(period, gap_positions, slot, random_state);
        },
        ctrl);
    // Only updates caught between the two back-to-back refclk reads are kept. Readings from the other two gaps span the
    // wall read or the block boundary, and measured up to 12 ns off under di/dt. Which gap catches an update is random,
    // so the kept readings still sample every moment equally.
    calibration.gap_positions = eth_clock::GapPositions::of(kDropGap, calibration.gap_positions.at(1), kDropGap);
    Ruler::Output output;
    Ruler::Update held[kHistory];
    Ruler ruler{output, held};
    if (ctrl->stop == 0u) {
        const eth_ptp::Instant start = eth_ptp::read_instant();
        uint64_t refclk = start.refclk, wall_eighths = start.wall << 3;
        uint32_t iter = 0;
        while (true) {
            uint32_t update[2];
            if (run(calibration.period, calibration.gap_positions, update, random_state) != 0) {
                refclk += update[0] - static_cast<uint32_t>(refclk);
                wall_eighths += update[1] - static_cast<uint32_t>(wall_eighths);
                ruler.on_update(refclk, wall_eighths);
            }
            if ((++iter & kStopPollMask) != 0u) {
                continue;
            }
            invalidate_l1_cache();
            if (ctrl->stop != 0u) {
                break;
            }
        }
        ruler.release_held();
    }
    output.writer.close();
    std::atomic_thread_fence(std::memory_order_release);
    ctrl->done = kernel_profiler::kResidentDoneWord;
}
