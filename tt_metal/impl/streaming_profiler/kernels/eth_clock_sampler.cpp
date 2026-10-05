// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"
#include "internal/ethernet/eth_ptp.hpp"
#include "tt_metal/impl/streaming_profiler/kernels/eth_clock.hpp"

namespace eth_clock {
// The stream's block length in cycles. Pads are drawn uniformly over this range so each pass starts at a uniform phase
// relative to the blocks.
constexpr uint32_t kPadRange = 6;
constexpr uint32_t kPadSlots = 256;
// The arguments to sampler_stream. The registers are passed in as data so the compiler keeps their addresses in
// registers instead of rebuilding them between blocks.
struct StreamArgs {
    eth_ptp::Reg<> refclk;
    eth_ptp::Reg<> wall;
    Calibration calibration;
    volatile uint32_t* slots;
    uint32_t mask;
    volatile uint32_t* head;
    uint32_t limit;
    bool stop_on_reject;
    const uint32_t* pads;
    volatile uint32_t* tail_word;
    uint32_t tail;

    // Sets up one calibration pass, which stores the first update it catches into `slot` and returns. *head must be
    // kSyncHeadStop so the reload after that update ends the call, and tail_word must point at a word the model never
    // reads. The fields are copied one by one because copying the structs goes through the stack at -Os.
    FORCE_INLINE void prepare_calibration_pass(
        uint32_t period, GapPositions gap_positions, volatile tt_l1_ptr uint32_t* slot) {
        calibration.period = period;
        calibration.gap_positions.word = gap_positions.word;
        slots = slot;
        mask = 0;
        stop_on_reject = true;
        limit = tail + 1;
    }
    // Sets up sampling, which streams every update into `ring` until the model stops it, with no limit on rejected
    // passes.
    FORCE_INLINE void prepare_sampling(
        const Calibration& calibrated, volatile tt_l1_ptr kernel_profiler::SyncSampleRing* ring) {
        calibration.period = calibrated.period;
        calibration.gap_positions.word = calibrated.gap_positions.word;
        slots = &ring->samples[0].refclk;
        mask = kernel_profiler::kSyncSampleRingSamples - 1;
        stop_on_reject = false;
        tail_word = &ring->tail;
        tail = 0;
        limit = mask;
    }
};

// Catches refclk updates into args.slots, starting at args.tail, with update n going to slot n & args.mask. Returns the
// tail after the last update it stored.
//
// Each pass first runs args.pads[tail % kPadSlots] / 4 nops, then reads blocks of refclk, refclk, wall, refclk until
// the refclk steps. Calibration and sampling passes therefore start at the same spread of phases.
//
// When args.calibration.period is nonzero, an update is stored only if its two blocks each took exactly period wall
// ticks. It is stored as the new refclk value and the wall time of the step in eighths, which is the block's wall read
// plus its gap's position from args.calibration.gap_positions. Each stored update also writes the previous tail to
// *args.tail_word, which publishes every earlier update, so a published sample was stored at least one handler earlier.
//
// When args.calibration.period is 0, for calibration, every update is stored as its block's wall length and nothing is
// published.
//
// At args.limit the stream reloads *args.head, the model's position, and moves the limit args.mask past it, waiting
// while the ring is full. It returns when the head is kSyncHeadStop past the model's position, or, with
// args.stop_on_reject, after a pass that stored nothing.
//
// The optimize attributes are needed because at -Os GCC puts the calibration path inline and merges the handlers'
// tails. The extra taken branches cost more with the branch predictor off, and made the stream miss every other update
// at 800 MHz.
__attribute__((noinline, aligned(64), optimize("no-crossjumping", "reorder-blocks-algorithm=stc"))) inline uint32_t
sampler_stream(const StreamArgs& args) {
    struct Block {
        uint32_t first_refclk, second_refclk, wall, last_refclk;
    };
    enum class Step : uint8_t { Stored, Refill, Rejected };
    const eth_ptp::Reg<> refclk = args.refclk, wall = args.wall;
    const uint32_t period = args.calibration.period, mask = args.mask;
    const GapPositions gap_positions{args.calibration.gap_positions.word};
    volatile uint32_t* const slots = args.slots;
    volatile uint32_t* const head = args.head;
    uint32_t limit = args.limit;
    const bool stop_on_reject = args.stop_on_reject;
    const uint32_t* const pads = args.pads;
    volatile uint32_t* const tail_word = args.tail_word;
    uint32_t tail = args.tail;
    const auto read = [&]() __attribute__((always_inline)) {
        Block block;
        block.first_refclk = refclk.read();
        block.second_refclk = refclk.read();
        block.wall = wall.read();
        block.last_refclk = refclk.read();
        return block;
    };
    // A pass jumps pad bytes back from the end of the nops. The label is defined in asm, so this function must only be
    // emitted once.
    uint32_t nops_end;
    asm("lla %0, .Lsampler_stream_nops_end" : "=r"(nops_end));
    uint32_t pad = pads[tail & (kPadSlots - 1)];
    disable_branch_predictor();
    // The pass starts on a cache line so its blocks have the same layout in every build. Passing pad through the asm
    // keeps its load ahead of the alignment, so nothing is placed between the alignment and the pass.
    asm volatile(".p2align 6" : "+r"(pad));
    // With odds this low, the compiler makes a step at the last check a single taken branch into its handler.
    const auto stepped = [](const Block& before, const Block& last) __attribute__((always_inline)) {
        return __builtin_expect_with_probability(last.last_refclk != before.last_refclk, 0, 0.001);
    };
    // A step that happened between before's and last's final reads is placed using last's reads. now gives the length
    // of the block after it.
    const auto store = [&](const Block& before, const Block& last, const Block& now)
                           __attribute__((always_inline)) -> Step {
        const uint32_t step_block_ticks = last.wall - before.wall, next_block_ticks = now.wall - last.wall;
        volatile uint32_t* const slot = slots + 2 * (tail & mask);
        if (__builtin_expect(period != 0, 1)) {
            if (((step_block_ticks ^ period) | (next_block_ticks ^ period)) != 0) {
                return Step::Rejected;
            }
            const uint32_t gap = static_cast<uint32_t>(last.first_refclk == before.last_refclk)
                                 << (last.second_refclk == last.first_refclk);
            slot[0] = last.last_refclk;
            slot[1] = (last.wall << 3) + static_cast<uint32_t>(gap_positions.at(gap));
            *tail_word = tail;
            return ++tail == limit ? Step::Refill : Step::Stored;
        }
        slot[0] = next_block_ticks;
        return ++tail == limit ? Step::Refill : Step::Stored;
    };
    // Reads blocks until the refclk steps, then stores the step. Rotating three named blocks keeps each check's reads
    // in the registers they were loaded into. A single rotating variable or an array makes the compiler copy them into
    // one handler's registers.
    const auto run = [&]() __attribute__((always_inline)) -> Step {
        Block a = read(), b = read(), c;
#pragma GCC unroll 6
        for (uint32_t i = 0; i < 6; i++) {
            c = read();
            if (stepped(a, b)) {
                return store(a, b, c);
            }
            a = read();
            if (stepped(b, c)) {
                return store(b, c, a);
            }
            b = read();
            if (stepped(c, a)) {
                return store(c, a, b);
            }
        }
        return Step::Rejected;
    };
pass:
    asm volatile(
        ".option push\n\t.option norvc\n\t"
        "sub t0, %[end], %[pad]\n\t"
        "jr t0\n\t"
        ".rept %[n]\n\tnop\n\t.endr\n"
        ".Lsampler_stream_nops_end:\n\t"
        ".option pop"
        :
        : [end] "r"(nops_end), [pad] "r"(pad), [n] "i"(kPadRange - 1)
        : "t0", "memory");
    pad = pads[tail & (kPadSlots - 1)];
    switch (run()) {
        case Step::Stored: goto pass;
        case Step::Refill: goto refill;
        case Step::Rejected: goto reject;
    }
reject:
    if (!stop_on_reject) {
        goto pass;
    }
    goto done;
refill:
    for (;;) {
        invalidate_l1_cache();
        const uint32_t model_head = *head;
        if (__builtin_expect(static_cast<int32_t>(tail - model_head) < 0, 0)) {
            goto done;
        }
        limit = model_head + mask;
        if (limit != tail) {
            break;
        }
    }
    pad = pads[tail & (kPadSlots - 1)];
    goto pass;
done:
    enable_branch_predictor();
    return tail;
}
}  // namespace eth_clock

constexpr uint32_t kCtrlAddr = get_named_compile_time_arg_val("ctrl");
constexpr uint32_t kSampleRingAddr = get_named_compile_time_arg_val("sample_ring");

void kernel_main() {
    volatile tt_l1_ptr kernel_profiler::ResidentCtrl* ctrl =
        reinterpret_cast<volatile tt_l1_ptr kernel_profiler::ResidentCtrl*>(kCtrlAddr);
    volatile tt_l1_ptr kernel_profiler::SyncSampleRing* ring =
        reinterpret_cast<volatile tt_l1_ptr kernel_profiler::SyncSampleRing*>(kSampleRingAddr);
    // The stream's pads in bytes (4 per nop), uniform over kPadRange nops. They are drawn once up front so each pass's
    // pad is a single load.
    static uint32_t pads[eth_clock::kPadSlots];
    uint32_t random_state = eth_ptp::kWallClockLo.read() | 1u;
    for (uint32_t& pad : pads) {
        pad = 4u * eth_clock::draw(random_state, eth_clock::kPadRange);
    }
    uint32_t unpublished = 0;
    eth_clock::StreamArgs args{
        .refclk = eth_ptp::kRefclkLo,
        .wall = eth_ptp::kWallClockLo,
        .head = &ring->head,
        .pads = pads,
        .tail_word = &unpublished};
    ring->head = kernel_profiler::kSyncHeadStop;
    const eth_clock::Calibration calibration = eth_clock::calibrate(
        [&](uint32_t period, eth_clock::GapPositions gap_positions, volatile tt_l1_ptr uint32_t* slot) {
            const uint32_t from = args.tail;
            args.prepare_calibration_pass(period, gap_positions, slot);
            args.tail = sampler_stream(args);
            return args.tail - from;
        },
        ctrl);
    if (ctrl->stop == 0u) {
        const eth_ptp::Instant start = eth_ptp::read_instant();
        ring->refclk = start.refclk;
        ring->wall_eighths = start.wall << 3;
        ring->head = 0;
        args.prepare_sampling(calibration, ring);
        const uint32_t tail = sampler_stream(args);
        std::atomic_thread_fence(std::memory_order_release);
        ring->tail = tail;
    }
    std::atomic_thread_fence(std::memory_order_release);
    ring->done = 1;
}
