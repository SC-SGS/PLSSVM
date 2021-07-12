#pragma once

namespace plssvm {

constexpr unsigned THREADBLOCK_SIZE = 16;   //TODO: immer nur 1 mal
constexpr unsigned INTERNALBLOCK_SIZE = 6;  // minimal 1
constexpr unsigned THREADS_PER_BLOCK = 1024;

}  // namespace plssvm