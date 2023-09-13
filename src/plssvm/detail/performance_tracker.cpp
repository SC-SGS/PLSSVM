/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/performance_tracker.hpp"

#include "plssvm/constants.hpp"                          // plssvm::real_type
#include "plssvm/detail/arithmetic_type_name.hpp"        // plssvm::detail::arithmetic_type_name_v
#include "plssvm/detail/arithmetic_type_name.hpp"        // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/assert.hpp"                      // PLSSVM_ASSERT, PLSSVM_ASSERT_ENABLED
#include "plssvm/detail/cmd/parser_predict.hpp"          // plssvm::detail::cmd::parser_predict
#include "plssvm/detail/cmd/parser_scale.hpp"            // plssvm::detail::cmd::parser_scale
#include "plssvm/detail/cmd/parser_train.hpp"            // plssvm::detail::cmd::parser_train
#include "plssvm/detail/string_conversion.hpp"           // plssvm::detail::split_as
#include "plssvm/detail/string_utility.hpp"              // plssvm::detail::trim
#include "plssvm/detail/utility.hpp"                     // plssvm::detail::current_date_time
#include "plssvm/parameter.hpp"                          // plssvm::parameter
#include "plssvm/version/git_metadata/git_metadata.hpp"  // plssvm::version::git_metadata::commit_sha1
#include "plssvm/version/version.hpp"                    // plssvm::version::{version, detail::target_platforms}

#include "fmt/chrono.h"   // format std::chrono types
#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // format types with an operator<< overload

#if __has_include(<unistd.h>)
    #include <unistd.h>  // gethostname, getlogin_r, HOST_NAME_MAX, LOGIN_NAME_MAX
    #define PLSSVM_UNISTD_AVAILABLE
#endif

#include <array>          // std::array
#include <fstream>        // std::ofstream
#include <iostream>       // std::ios_base::app
#include <memory>         // std::shared_ptr
#include <string>         // std::string
#include <unordered_map>  // std::unordered_multimap
#include <utility>        // std::pair

namespace plssvm::detail {

performance_tracker::performance_tracker() {}
performance_tracker::~performance_tracker() {}

void performance_tracker::add_tracking_entry(const tracking_entry<std::string> &entry) {
    tracking_statistics.emplace(entry.entry_category, fmt::format("{}{}: \"{}\"\n", entry.entry_category.empty() ? "" : "  ", entry.entry_name, entry.entry_value));
}

void performance_tracker::add_tracking_entry(const tracking_entry<plssvm::parameter> &entry) {
    if (is_tracking()) {
        tracking_statistics.emplace("parameter", fmt::format("  kernel_type: {}\n"
                                                             "  degree:      {}\n"
                                                             "  gamma:       {}\n"
                                                             "  coef0:       {}\n"
                                                             "  cost:        {}\n"
                                                             "  real_type:   {}\n",
                                                             entry.entry_value.kernel_type.value(),
                                                             entry.entry_value.degree.value(),
                                                             entry.entry_value.gamma.is_default() ? std::string{ "#data_points" } : fmt::format("{}", entry.entry_value.gamma.value()),
                                                             entry.entry_value.coef0.value(),
                                                             entry.entry_value.cost.value(),
                                                             arithmetic_type_name<real_type>()));
    }
}

void performance_tracker::add_tracking_entry(const tracking_entry<cmd::parser_train> &entry) {
    if (is_tracking()) {
        tracking_statistics.emplace(entry.entry_category, fmt::format("  task:                        train\n"
                                                                      "  kernel_type:                 {}\n"
                                                                      "  degree:                      {}\n"
                                                                      "  gamma:                       {}\n"
                                                                      "  coef0:                       {}\n"
                                                                      "  cost:                        {}\n"
                                                                      "  epsilon:                     {}\n"
                                                                      "  max_iter:                    {}\n"
                                                                      "  classification_type:         {}\n"
                                                                      "  backend:                     {}\n"
                                                                      "  target:                      {}\n"
                                                                      "  sycl_kernel_invocation_type: {}\n"
                                                                      "  sycl_implementation_type:    {}\n"
                                                                      "  strings_as_labels:           {}\n"
                                                                      "  real_type:                   {}\n"
                                                                      "  input_filename:              \"{}\"\n"
                                                                      "  model_filename:              \"{}\"\n",
                                                                      entry.entry_value.csvm_params.kernel_type.value(),
                                                                      entry.entry_value.csvm_params.degree.value(),
                                                                      entry.entry_value.csvm_params.gamma.value(),
                                                                      entry.entry_value.csvm_params.coef0.value(),
                                                                      entry.entry_value.csvm_params.cost.value(),
                                                                      entry.entry_value.epsilon.value(),
                                                                      entry.entry_value.max_iter.value(),
                                                                      entry.entry_value.classification.value(),
                                                                      entry.entry_value.backend,
                                                                      entry.entry_value.target,
                                                                      entry.entry_value.sycl_kernel_invocation_type,
                                                                      entry.entry_value.sycl_implementation_type,
                                                                      entry.entry_value.strings_as_labels,
                                                                      arithmetic_type_name<real_type>(),
                                                                      entry.entry_value.input_filename,
                                                                      entry.entry_value.model_filename));
    }
}

void performance_tracker::add_tracking_entry(const tracking_entry<cmd::parser_predict> &entry) {
    if (is_tracking()) {
        tracking_statistics.emplace(entry.entry_category, fmt::format("  task:                     predict\n"
                                                                      "  backend:                  {}\n"
                                                                      "  target:                   {}\n"
                                                                      "  sycl_implementation_type: {}\n"
                                                                      "  strings_as_labels:        {}\n"
                                                                      "  real_type:                {}\n"
                                                                      "  input_filename:           \"{}\"\n"
                                                                      "  model_filename:           \"{}\"\n"
                                                                      "  predict_filename:         \"{}\"\n",
                                                                      entry.entry_value.backend,
                                                                      entry.entry_value.target,
                                                                      entry.entry_value.sycl_implementation_type,
                                                                      entry.entry_value.strings_as_labels,
                                                                      arithmetic_type_name<real_type>(),
                                                                      entry.entry_value.input_filename,
                                                                      entry.entry_value.model_filename,
                                                                      entry.entry_value.predict_filename));
    }
}

void performance_tracker::add_tracking_entry(const tracking_entry<cmd::parser_scale> &entry) {
    if (is_tracking()) {
        tracking_statistics.emplace(entry.entry_category, fmt::format("  task:               scale\n"
                                                                      "  lower:              {}\n"
                                                                      "  upper:              {}\n"
                                                                      "  format:             {}\n"
                                                                      "  strings_as_labels:  {}\n"
                                                                      "  real_type:          {}\n"
                                                                      "  input_filename:     \"{}\"\n"
                                                                      "  scaled_filename:    \"{}\"\n"
                                                                      "  save_filename:      \"{}\"\n"
                                                                      "  restore_filename:   \"{}\"\n",
                                                                      entry.entry_value.lower,
                                                                      entry.entry_value.upper,
                                                                      entry.entry_value.format,
                                                                      entry.entry_value.strings_as_labels,
                                                                      arithmetic_type_name<real_type>(),
                                                                      entry.entry_value.input_filename,
                                                                      entry.entry_value.scaled_filename,
                                                                      entry.entry_value.save_filename,
                                                                      entry.entry_value.restore_filename));
    }
}

void performance_tracker::save(const std::string &filename) {
    if (filename.empty()) {
        // write tracking entries to std::clog
        // NOTE: the tracking entries are always dumped to stdout, even if the --quiet flag has been provided
        std::clog << std::endl;
        save(std::clog);
    } else {
        // write the tracking entries to the specified file
        std::ofstream out{ filename, std::ios_base::app };
        save(out);
    }
}

void performance_tracker::save(std::ostream &out) {
    // append the current performance statistics to an already existing file if possible
    PLSSVM_ASSERT(out.good(), "Can't write to the provided output stream!");

    // get the current host- and username
#if defined(PLSSVM_UNISTD_AVAILABLE)
    std::string hostname(HOST_NAME_MAX, '\0');
    gethostname(hostname.data(), HOST_NAME_MAX);
    std::string username(LOGIN_NAME_MAX, '\0');
    getlogin_r(username.data(), LOGIN_NAME_MAX);
#else
    constexpr std::string_view hostname{ "not available" };
    constexpr std::string_view username{ "not available" };
#endif
    // check whether asserts are enabled
#if defined(PLSSVM_ASSERT_ENABLED)
    constexpr bool assert_enabled = true;
#else
    constexpr bool assert_enabled = false;
#endif
    // check whether LTO has been enabled
#if defined(PLSSVM_LTO_SUPPORTED)
    constexpr bool lto_enabled = true;
#else
    constexpr bool lto_enabled = false;
#endif
    // check whether GEMM has been used instead of SYMM
#if defined(PLSSVM_USE_GEMM)
    constexpr bool use_gemm = true;
#else
    constexpr bool use_gemm = false;
#endif
    // check whether the maximum memory allocation size for the plssvm::solver_type::automatic has been enforced
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
    constexpr bool enforce_max_mem_alloc_size = true;
#else
    constexpr bool enforce_max_mem_alloc_size = false;
#endif


    // begin a new YAML document (only with "---" multiple YAML docments in a single file are allowed)
    out << "---\n";

    // output metadata information
    out << fmt::format(
        "meta_data:\n"
        "  date:                       \"{}\"\n"
        "  PLSSVM_TARGET_PLATFORMS:    \"{}\"\n"
        "  commit:                     {}\n"
        "  version:                    {}\n"
        "  hostname:                   {}\n"
        "  user:                       {}\n"
        "  build_type:                 {}\n"
        "  LTO:                        {}\n"
        "  asserts:                    {}\n"
        "  gemm:                       {}\n"
        "  enforce_max_mem_alloc_size: {}\n"
        "  PLSSVM_THREAD_BLOCK_SIZE:   {}\n"
        "  PLSSVM_INTERNAL_BLOCK_SIZE: {}\n"
        "  PLSSVM_OPENMP_BLOCK_SIZE:   {}\n",
        plssvm::detail::current_date_time(),
        version::detail::target_platforms,
        version::git_metadata::commit_sha1().empty() ? "unknown" : version::git_metadata::commit_sha1(),
        version::version,
        hostname.data(),
        username.data(),
        PLSSVM_BUILD_TYPE,
        lto_enabled,
        assert_enabled,
        use_gemm,
        enforce_max_mem_alloc_size,
        THREAD_BLOCK_SIZE,
        INTERNAL_BLOCK_SIZE,
        OPENMP_BLOCK_SIZE);

#if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    //  check whether DPC++ AOT has been enabled
    #if defined(PLSSVM_SYCL_BACKEND_DPCPP_ENABLE_AOT)
    constexpr bool dpcpp_aot = true;
    #else
    constexpr bool dpcpp_aot = false;
    #endif

    out << fmt::format(
        "  DPCPP_backend_type:         {}\n"
        "  DPCPP_amd_gpu_backend_type: {}\n"
        "  DPCPP_with_aot:             {}\n",
        PLSSVM_SYCL_BACKEND_DPCPP_BACKEND_TYPE,
        PLSSVM_SYCL_BACKEND_DPCPP_GPU_AMD_BACKEND_TYPE,
        dpcpp_aot);
#endif
#if defined(PLSSVM_SYCL_BACKEND_HAS_HIPSYCL)
    // check whether hipSYCL's new SSCP has been enabled
    #if defined(PLSSVM_SYCL_BACKEND_HIPSYCL_USE_GENERIC_SSCP)
    constexpr bool hipsycl_sscp = true;
    #else
    constexpr bool hipsycl_sscp = false;
    #endif

    out << fmt::format(
        "  HIPSYCL_with_generic_SSCP:  {}\n",
        hipsycl_sscp);
#endif
    out << "\n";

    // output the actual (performance) statistics
    std::unordered_multimap<std::string, std::string>::iterator group_iter;  // iterate over all groups
    std::unordered_multimap<std::string, std::string>::iterator entry_iter;  // iterate over all entries in a specific group
    for (group_iter = tracking_statistics.begin(); group_iter != tracking_statistics.end(); group_iter = entry_iter) {
        // get the current group
        const std::string &group = group_iter->first;
        // find the range of all entries in the current group
        const std::pair key_range = tracking_statistics.equal_range(group);

        // output the group name, if it is not the empty string
        if (!group.empty()) {
            out << group << ":\n";
        }
        // regroup multiple entries into single arrays
        if (group == "cg") {
            // special case: multiple conjugate gradients could be performed (e.g., in OAO)
            std::unordered_map<std::string, std::vector<std::string>> cg_map{};
            // group the same categories into vectors
            for (entry_iter = key_range.first; entry_iter != key_range.second; ++entry_iter) {
                const std::vector<std::string> split = detail::split_as<std::string>(detail::trim(entry_iter->second), ' ');
                if (cg_map.count(split[0]) == 0) {
                    cg_map.emplace(split[0], std::vector<std::string>{});
                }
                cg_map[split[0]].emplace_back(split[1].data(), split[1].size() - 1);
            }
            // output all CG categories
            for (const auto &[cg_category, cg_values] : cg_map) {
                if (cg_values.size() == 1) {
                    out << fmt::format("  {} {}\n", cg_category, cg_values.front());
                } else {
                    out << fmt::format("  {} [{}]\n", cg_category, fmt::join(cg_values, ","));
                }
            }
        } else {
            // output all performance statistic entries of the current group
            for (entry_iter = key_range.first; entry_iter != key_range.second; ++entry_iter) {
                out << fmt::format("{}", entry_iter->second);
            }
        }

        out << '\n';
    }

    // clear tracking statistics
    tracking_statistics.clear();
}

void performance_tracker::pause_tracking() noexcept { is_tracking_ = false; }
void performance_tracker::resume_tracking() noexcept { is_tracking_ = true; }
bool performance_tracker::is_tracking() noexcept { return is_tracking_; }
const std::unordered_multimap<std::string, std::string> &performance_tracker::get_tracking_entries() noexcept { return tracking_statistics; }

std::shared_ptr<performance_tracker> global_tracker = std::make_shared<performance_tracker>();

}  // namespace plssvm::detail