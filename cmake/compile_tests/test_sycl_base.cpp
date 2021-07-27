#include "sycl/sycl.hpp"

int main() {
// TODO: check necessary features?
#if defined(SYCL_LANGUAGE_VERSION)
    static_assert(SYCL_LANGUAGE_VERSION >= 202001, "Insufficient SYCL language version!");
#endif
    return 0;
}
