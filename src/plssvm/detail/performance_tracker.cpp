/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/performance_tracker.hpp"

#include "plssvm/constants.hpp"                          // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/detail/arithmetic_type_name.hpp"        // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/assert.hpp"                      // PLSSVM_ASSERT, PLSSVM_ASSERT_ENABLED
#include "plssvm/detail/cmd/parser_predict.hpp"          // plssvm::detail::cmd::parser_predict
#include "plssvm/detail/cmd/parser_scale.hpp"            // plssvm::detail::cmd::parser_scale
#include "plssvm/detail/cmd/parser_train.hpp"            // plssvm::detail::cmd::parser_train
#include "plssvm/detail/utility.hpp"                     // plssvm::detail::current_date_time, PLSSVM_IS_DEFINED
#include "plssvm/parameter.hpp"                          // plssvm::parameter
#include "plssvm/version/git_metadata/git_metadata.hpp"  // plssvm::version::git_metadata::commit_sha1
#include "plssvm/version/version.hpp"                    // plssvm::version::{version, detail::target_platforms}

#include "fmt/chrono.h"   // format std::chrono types
#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // format types with an operator<< overload

#if __has_include(<unistd.h>) && __has_include(<limits.h>)
    #include <limits.h>  // HOST_NAME_MAX, LOGIN_NAME_MAX
    #include <unistd.h>  // gethostname, getlogin_r,
    #define PLSSVM_UNISTD_AVAILABLE
#endif

#include <algorithm>    // std::max
#include <cstddef>      // std::size_t
#include <fstream>      // std::ofstream
#include <iostream>     // std::ios_base::app, std::ostream, std::clog, std::endl
#include <map>          // std::map
#include <memory>       // std::shared_ptr
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace plssvm::detail {

performance_tracker::performance_tracker() = default;
performance_tracker::~performance_tracker() = default;

void performance_tracker::add_tracking_entry(const tracking_entry<plssvm::parameter> &entry) {
    if (is_tracking()) {
        // create category
        tracking_statistics_.emplace(entry.entry_category, std::map<std::string, std::vector<std::string>>{});
        // fill category with value
        tracking_statistics_["parameter"].emplace("kernel_type", std::vector<std::string>{ fmt::format("{}", entry.entry_value.kernel_type.value()) });
        tracking_statistics_["parameter"].emplace("degree", std::vector<std::string>{ fmt::format("{}", entry.entry_value.degree.value()) });
        tracking_statistics_["parameter"].emplace("gamma", std::vector<std::string>{ entry.entry_value.gamma.is_default() ? std::string{ "#data_points" } : fmt::format("{}", entry.entry_value.gamma.value()) });
        tracking_statistics_["parameter"].emplace("coef0", std::vector<std::string>{ fmt::format("{}", entry.entry_value.coef0.value()) });
        tracking_statistics_["parameter"].emplace("cost", std::vector<std::string>{ fmt::format("{}", entry.entry_value.cost.value()) });
        tracking_statistics_["parameter"].emplace("real_type", std::vector<std::string>{ std::string{ arithmetic_type_name<real_type>() } });
    }
}

void performance_tracker::add_tracking_entry(const tracking_entry<cmd::parser_train> &entry) {
    if (is_tracking()) {
        // create category
        tracking_statistics_.emplace(entry.entry_category, std::map<std::string, std::vector<std::string>>{});
        // fill category with value
        tracking_statistics_[entry.entry_category].emplace("task", std::vector<std::string>{ "train" });
        tracking_statistics_[entry.entry_category].emplace("kernel_type", std::vector<std::string>{ fmt::format("{}", entry.entry_value.csvm_params.kernel_type.value()) });
        tracking_statistics_[entry.entry_category].emplace("degree", std::vector<std::string>{ fmt::format("{}", entry.entry_value.csvm_params.degree.value()) });
        tracking_statistics_[entry.entry_category].emplace("gamma", std::vector<std::string>{ fmt::format("{}", entry.entry_value.csvm_params.gamma.value()) });
        tracking_statistics_[entry.entry_category].emplace("coef0", std::vector<std::string>{ fmt::format("{}", entry.entry_value.csvm_params.coef0.value()) });
        tracking_statistics_[entry.entry_category].emplace("cost", std::vector<std::string>{ fmt::format("{}", entry.entry_value.csvm_params.cost.value()) });
        tracking_statistics_[entry.entry_category].emplace("epsilon", std::vector<std::string>{ fmt::format("{}", entry.entry_value.epsilon.value()) });
        tracking_statistics_[entry.entry_category].emplace("max_iter", std::vector<std::string>{ fmt::format("{}", entry.entry_value.max_iter.value()) });
        tracking_statistics_[entry.entry_category].emplace("classification_type", std::vector<std::string>{ fmt::format("{}", entry.entry_value.classification.value()) });
        tracking_statistics_[entry.entry_category].emplace("backend", std::vector<std::string>{ fmt::format("{}", entry.entry_value.backend) });
        tracking_statistics_[entry.entry_category].emplace("target", std::vector<std::string>{ fmt::format("{}", entry.entry_value.target) });
        tracking_statistics_[entry.entry_category].emplace("sycl_kernel_invocation_type", std::vector<std::string>{ fmt::format("{}", entry.entry_value.sycl_kernel_invocation_type) });
        tracking_statistics_[entry.entry_category].emplace("sycl_implementation_type", std::vector<std::string>{ fmt::format("{}", entry.entry_value.sycl_implementation_type) });
        tracking_statistics_[entry.entry_category].emplace("strings_as_labels", std::vector<std::string>{ fmt::format("{}", entry.entry_value.strings_as_labels) });
        tracking_statistics_[entry.entry_category].emplace("real_type", std::vector<std::string>{ std::string{ arithmetic_type_name<real_type>() } });
        tracking_statistics_[entry.entry_category].emplace("input_filename", std::vector<std::string>{ fmt::format("\"{}\"", entry.entry_value.input_filename) });
        tracking_statistics_[entry.entry_category].emplace("model_filename", std::vector<std::string>{ fmt::format("\"{}\"", entry.entry_value.model_filename) });
    }
}

void performance_tracker::add_tracking_entry(const tracking_entry<cmd::parser_predict> &entry) {
    if (is_tracking()) {
        // create category
        tracking_statistics_.emplace(entry.entry_category, std::map<std::string, std::vector<std::string>>{});
        // fill category with value
        tracking_statistics_[entry.entry_category].emplace("task", std::vector<std::string>{ "predict" });
        tracking_statistics_[entry.entry_category].emplace("backend", std::vector<std::string>{ fmt::format("{}", entry.entry_value.backend) });
        tracking_statistics_[entry.entry_category].emplace("target", std::vector<std::string>{ fmt::format("{}", entry.entry_value.target) });
        tracking_statistics_[entry.entry_category].emplace("sycl_implementation_type", std::vector<std::string>{ fmt::format("{}", entry.entry_value.sycl_implementation_type) });
        tracking_statistics_[entry.entry_category].emplace("strings_as_labels", std::vector<std::string>{ fmt::format("{}", entry.entry_value.strings_as_labels) });
        tracking_statistics_[entry.entry_category].emplace("real_type", std::vector<std::string>{ std::string{ arithmetic_type_name<real_type>() } });
        tracking_statistics_[entry.entry_category].emplace("input_filename", std::vector<std::string>{ fmt::format("\"{}\"", entry.entry_value.input_filename) });
        tracking_statistics_[entry.entry_category].emplace("model_filename", std::vector<std::string>{ fmt::format("\"{}\"", entry.entry_value.model_filename) });
        tracking_statistics_[entry.entry_category].emplace("predict_filename", std::vector<std::string>{ fmt::format("\"{}\"", entry.entry_value.predict_filename) });
    }
}

void performance_tracker::add_tracking_entry(const tracking_entry<cmd::parser_scale> &entry) {
    if (is_tracking()) {
        // create category
        tracking_statistics_.emplace(entry.entry_category, std::map<std::string, std::vector<std::string>>{});
        // fill category with value
        tracking_statistics_[entry.entry_category].emplace("task", std::vector<std::string>{ "scale" });
        tracking_statistics_[entry.entry_category].emplace("lower", std::vector<std::string>{ fmt::format("{}", entry.entry_value.lower) });
        tracking_statistics_[entry.entry_category].emplace("upper", std::vector<std::string>{ fmt::format("{}", entry.entry_value.upper) });
        tracking_statistics_[entry.entry_category].emplace("format", std::vector<std::string>{ fmt::format("{}", entry.entry_value.format) });
        tracking_statistics_[entry.entry_category].emplace("strings_as_labels", std::vector<std::string>{ fmt::format("{}", entry.entry_value.strings_as_labels) });
        tracking_statistics_[entry.entry_category].emplace("real_type", std::vector<std::string>{ std::string{ arithmetic_type_name<real_type>() } });
        tracking_statistics_[entry.entry_category].emplace("input_filename", std::vector<std::string>{ fmt::format("\"{}\"", entry.entry_value.input_filename) });
        tracking_statistics_[entry.entry_category].emplace("scaled_filename", std::vector<std::string>{ fmt::format("\"{}\"", entry.entry_value.scaled_filename) });
        tracking_statistics_[entry.entry_category].emplace("save_filename", std::vector<std::string>{ fmt::format("\"{}\"", entry.entry_value.save_filename) });
        tracking_statistics_[entry.entry_category].emplace("restore_filename", std::vector<std::string>{ fmt::format("\"{}\"", entry.entry_value.restore_filename) });
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
    constexpr bool assert_enabled = PLSSVM_IS_DEFINED(PLSSVM_ASSERT_ENABLED);
    // check whether LTO has been enabled
    constexpr bool lto_enabled = PLSSVM_IS_DEFINED(PLSSVM_LTO_SUPPORTED);
    // check whether GEMM has been used instead of SYMM
    constexpr bool use_gemm = PLSSVM_IS_DEFINED(PLSSVM_USE_GEMM);
    // check whether the maximum allocatable memory size should be enforced
    constexpr bool enforce_max_mem_alloc_size = PLSSVM_IS_DEFINED(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE);

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
        "  THREAD_BLOCK_SIZE:          {}\n"
        "  FEATURE_BLOCK_SIZE:         {}\n"
        "  INTERNAL_BLOCK_SIZE:        {}\n"
        "  PADDING_SIZE:               {}\n",
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
        FEATURE_BLOCK_SIZE,
        INTERNAL_BLOCK_SIZE,
        PADDING_SIZE);

#if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    //  check whether DPC++ AOT has been enabled
    constexpr bool dpcpp_aot = PLSSVM_IS_DEFINED(PLSSVM_SYCL_BACKEND_DPCPP_ENABLE_AOT);

    out << fmt::format(
        "  DPCPP_backend_type:         {}\n"
        "  DPCPP_amd_gpu_backend_type: {}\n"
        "  DPCPP_with_aot:             {}\n",
        PLSSVM_SYCL_BACKEND_DPCPP_BACKEND_TYPE,
        PLSSVM_SYCL_BACKEND_DPCPP_GPU_AMD_BACKEND_TYPE,
        dpcpp_aot);
#endif
#if defined(PLSSVM_SYCL_BACKEND_HAS_ADAPTIVECPP)
    // check whether AdaptiveCpp's new SSCP has been enabled
    constexpr bool adaptivecpp_sscp = PLSSVM_IS_DEFINED(PLSSVM_SYCL_BACKEND_ADAPTIVECPP_USE_GENERIC_SSCP);

    out << fmt::format(
        "  ADAPTIVECPP_with_generic_SSCP:  {}\n",
        adaptivecpp_sscp);
#endif
    out << "\n";

    // output the actual (performance) statistics
    for (const auto &[category, category_entries] : tracking_statistics_) {
        // output the category name if it isn't the empty string
        if (!category.empty()) {
            out << category << ":\n";
        }

        // calculate the number of padding whitespaces for this category
        std::size_t max_entry_name_length = 0;
        for (const auto &[category_entry, entries] : category_entries) {
            max_entry_name_length = std::max(max_entry_name_length, category_entry.size());
        }

        // output all entries in this category
        for (const auto &[category_entry, entries] : category_entries) {
            // output the name of the current category entry
            out << fmt::format("{}{}:{:>{}}", category.empty() ? "" : "  ", category_entry, "", max_entry_name_length - category_entry.size() + 1);
            // output all entry values
            if (entries.size() == 1) {
                // output only a single element
                out << entries.front();
            } else {
                // output multiple values as YAML array
                out << fmt::format("[{}]", fmt::join(entries, ", "));
            }
            out << '\n';
        }
        out << '\n';
    }

    // clear tracking statistics
    this->clear_tracking_entries();
}

void performance_tracker::pause_tracking() noexcept { is_tracking_ = false; }
void performance_tracker::resume_tracking() noexcept { is_tracking_ = true; }
bool performance_tracker::is_tracking() const noexcept { return is_tracking_; }
const std::map<std::string, std::map<std::string, std::vector<std::string>>> &performance_tracker::get_tracking_entries() noexcept { return tracking_statistics_; }
void performance_tracker::clear_tracking_entries() noexcept { tracking_statistics_.clear(); }

std::shared_ptr<performance_tracker> global_tracker = std::make_shared<performance_tracker>();

}  // namespace plssvm::detail

#undef PLSSVM_UNISTD_AVAILABLE