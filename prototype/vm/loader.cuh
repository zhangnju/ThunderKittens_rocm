#pragma once

#include "kittens.cuh"
#include "../common/common.cuh"
#include "util.cuh"

MAKE_WORKER(loader, TEVENT_LOADER_START, false)

MAKE_WORKER(sync_loader, TEVENT_SYNC_LOADER_START, false)

MAKE_WORKER(prefetcher, TEVENT_PREFETCHER_START, false)
