// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The frame and result layout in the eth core's L1, shared by ActiveEthPtpStamps and its kernel.

#pragma once

#include <cstdint>

namespace eth_ptp_stamps {

constexpr uint32_t kFrameBytes = 96;
// Offsets from the kernel's L1 base, which holds the 16-byte start and stop handshake.
constexpr uint32_t kFrameOffset = 64;
constexpr uint32_t kResultOffset = 512;
constexpr uint32_t kRounds = 256;
constexpr uint32_t kTwoStepFrames = 16;
constexpr uint32_t kDone = 0xD0E5u;

// The stamps, in PTP ns, of the frame this end received in one round. peer_egress is the peer's egress stamp, which the
// frame carries, and ingress is this end's ingress stamp.
struct RoundStamps {
    uint64_t peer_egress;
    uint64_t ingress;
};

struct Result {
    uint32_t done;
    uint32_t header_select_before, header_select_after;
    uint32_t no_match_before, no_match_after;
    // PTP ns minus refclk ns minus restart_ptp_timer's offset, read just after a refclk update.
    int32_t restart_error_ns;
    uint32_t unstamped;
    // Frames whose ingress stamp wasn't the FIFO's only entry with the stamp rule's label.
    uint32_t ingress_mismatched;
    uint32_t rounds;
    // The transmitter's two-step frames whose FIFO entry carried their tag and a time inside their send.
    uint32_t two_step_good;
    RoundStamps stamps[kRounds];
};

}  // namespace eth_ptp_stamps
