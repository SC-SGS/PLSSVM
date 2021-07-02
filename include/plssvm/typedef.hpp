#pragma once

namespace plssvm {

using real_t = double;  //TODO: float Fehler finden

static constexpr unsigned CUDABLOCK_SIZE = 16;                // minimal 1
static constexpr unsigned THREADBLOCK_SIZE = CUDABLOCK_SIZE;  //TODO: immer nur 1 mal
static constexpr unsigned INTERNALBLOCK_SIZE = 6;             // minimal 1
static constexpr unsigned BLOCKING_SIZE_THREAD = INTERNALBLOCK_SIZE;
static constexpr unsigned THREADS_PER_BLOCK = 1024;

}  // namespace plssvm