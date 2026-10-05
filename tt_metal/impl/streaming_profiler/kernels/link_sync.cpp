// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "eth_l1_address_map.h"
#include "internal/ethernet/dataflow_api.h"
#include "tt_metal/impl/streaming_profiler/kernels/link_sync.hpp"

constexpr bool kTransmitter = get_compile_time_arg_val(0) != 0;
static link_sync::LinkEnd<kTransmitter> g_link;
static constexpr uint32_t kHandshake = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
static constexpr uint32_t kHandshakeBytes = 16;
// The base firmware shares ERISC0 and only runs when a kernel yields to it. The fabric router yields every 10000 idle
// loops by default, and this kernel does the same.
static constexpr uint32_t kTurnsPerYield = 10000;

void kernel_main() {
    g_link.open(get_arg_val<uint32_t>(0));
    if constexpr (kTransmitter) {
        eth_send_bytes(kHandshake, kHandshake, kHandshakeBytes);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(kHandshakeBytes);
        eth_receiver_channel_done(0);
    }
    g_link.start();
    for (uint32_t turn = 1; g_link.l1->ctl != kernel_profiler::LinkSyncCtl::Stop; turn++) {
        if (g_link.due()) {
            g_link.serve();
        }
        if (turn % kTurnsPerYield == 0) {
            run_routing();
        }
    }
    g_link.stop();
    g_link.l1->done = kernel_profiler::kResidentDoneWord;
}
