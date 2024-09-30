/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/performance_tracker.hpp"

#include "plssvm/constants.hpp"                          // plssvm::real_type, plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE, plssvm::FEATURE_BLOCK_SIZE, plssvm::PADDING_SIZE
#include "plssvm/detail/arithmetic_type_name.hpp"        // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/assert.hpp"                      // PLSSVM_ASSERT, PLSSVM_ASSERT_ENABLED
#include "plssvm/detail/cmd/parser_predict.hpp"          // plssvm::detail::cmd::parser_predict
#include "plssvm/detail/cmd/parser_scale.hpp"            // plssvm::detail::cmd::parser_scale
#include "plssvm/detail/cmd/parser_train.hpp"            // plssvm::detail::cmd::parser_train
#include "plssvm/detail/string_utility.hpp"              // plssvm::detail::replace_all
#include "plssvm/detail/utility.hpp"                     // plssvm::detail::current_date_time, PLSSVM_IS_DEFINED
#include "plssvm/gamma.hpp"                              // plssvm::get_gamma_string
#include "plssvm/parameter.hpp"                          // plssvm::parameter
#include "plssvm/version/git_metadata/git_metadata.hpp"  // plssvm::version::git_metadata::commit_sha1
#include "plssvm/version/version.hpp"                    // plssvm::version::{version, detail::target_platforms}

#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
    #include "plssvm/detail/tracking/utility.hpp"

    #include "hardware_sampling/hardware_sampler.hpp"         // hws::hardware_sampler
    #include "hardware_sampling/system_hardware_sampler.hpp"  // hws::system_hardware_sampler
    #include "hardware_sampling/version.hpp"                  // hws::version::version
#endif

#include "cxxopts.hpp"   // CXXOPTS__VERSION_MAJOR, CXXOPTS__VERSION_MINOR, CXXOPTS__VERSION_MINOR
#include "fmt/base.h"    // FMT_VERSION
#include "fmt/chrono.h"  // format std::chrono types
#include "fmt/format.h"  // fmt::format
#include "fmt/ranges.h"  // fmt::join

#if __has_include(<unistd.h>)
    #include <unistd.h>  // gethostname, getlogin_r, sysconf, _SC_HOST_NAME_MAX, _SC_LOGIN_NAME_MAX
    #define PLSSVM_UNISTD_AVAILABLE
#endif

#if defined(PLSSVM_STDPAR_BACKEND_HAS_GNU_TBB)
    #include "boost/version.hpp"  // BOOST_VERSION
#endif

#if defined(PLSSVM_STDPAR_BACKEND_HAS_INTEL_LLVM)
    #include "oneapi/dpl/pstl/onedpl_config.h"  // ONEDPL_VERSION_MAJOR, ONEDPL_VERSION_MINOR, ONEDPL_VERSION_PATCH
#endif

#if defined(PLSSVM_STDPAR_BACKEND_HAS_ACPP) || defined(PLSSVM_STDPAR_BACKEND_HAS_GNU_TBB)
    #if __has_include("tbb/tbb_stddef.h")
        #include "tbb/tbb_stddef.h"  // TBB_VERSION_MAJOR, TBB_VERSION_MINOR
    #elif __has_include("tbb/version.h")
        #include "tbb/version.h"  // TBB_VERSION_MAJOR, TBB_VERSION_MINOR
    #else
    // no appropriate header found -> set version to 0
        #define TBB_VERSION_MAJOR 0
        #define TBB_VERSION_MINOR 0
    #endif
#endif

#include <algorithm>    // std::max
#include <chrono>       // std::chrono::steady_clock::time_point
#include <cstddef>      // std::size_t
#include <fstream>      // std::ofstream
#include <iostream>     // std::ios_base::app, std::ostream, std::clog, std::endl
#include <map>          // std::map
#include <memory>       // std::unique_ptr
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::move
#include <vector>       // std::vector

namespace plssvm::detail::tracking {

// Must be explicitly defaulted in the cpp file to prevent linker errors!
performance_tracker::performance_tracker() = default;
performance_tracker::performance_tracker(const performance_tracker &) = default;
performance_tracker::performance_tracker(performance_tracker &&) noexcept = default;
performance_tracker &performance_tracker::operator=(const performance_tracker &) = default;
performance_tracker &performance_tracker::operator=(performance_tracker &&) noexcept = default;
performance_tracker::~performance_tracker() = default;

void performance_tracker::add_tracking_entry(const tracking_entry<plssvm::parameter> &entry) {
    // check whether entries should currently be tracked
    if (this->is_tracking()) {
        // create category
        tracking_entries_.emplace(entry.entry_category, std::map<std::string, std::vector<std::string>>{});
        // fill category with value
        tracking_entries_["parameter"].emplace("kernel_type", std::vector<std::string>{ fmt::format("{}", entry.entry_value.kernel_type) });
        tracking_entries_["parameter"].emplace("degree", std::vector<std::string>{ fmt::format("{}", entry.entry_value.degree) });
        tracking_entries_["parameter"].emplace("gamma", std::vector<std::string>{ get_gamma_string(entry.entry_value.gamma) });
        tracking_entries_["parameter"].emplace("coef0", std::vector<std::string>{ fmt::format("{}", entry.entry_value.coef0) });
        tracking_entries_["parameter"].emplace("cost", std::vector<std::string>{ fmt::format("{}", entry.entry_value.cost) });
        tracking_entries_["parameter"].emplace("real_type", std::vector<std::string>{ std::string{ arithmetic_type_name<real_type>() } });
    }
}

void performance_tracker::add_tracking_entry(const tracking_entry<cmd::parser_train> &entry) {
    // check whether entries should currently be tracked
    if (this->is_tracking()) {
        // create category
        tracking_entries_.emplace(entry.entry_category, std::map<std::string, std::vector<std::string>>{});
        // fill category with value
        tracking_entries_[entry.entry_category].emplace("task", std::vector<std::string>{ "train" });
        tracking_entries_[entry.entry_category].emplace("kernel_type", std::vector<std::string>{ fmt::format("{}", entry.entry_value.csvm_params.kernel_type) });
        tracking_entries_[entry.entry_category].emplace("degree", std::vector<std::string>{ fmt::format("{}", entry.entry_value.csvm_params.degree) });
        tracking_entries_[entry.entry_category].emplace("gamma", std::vector<std::string>{ get_gamma_string(entry.entry_value.csvm_params.gamma) });
        tracking_entries_[entry.entry_category].emplace("coef0", std::vector<std::string>{ fmt::format("{}", entry.entry_value.csvm_params.coef0) });
        tracking_entries_[entry.entry_category].emplace("cost", std::vector<std::string>{ fmt::format("{}", entry.entry_value.csvm_params.cost) });
        tracking_entries_[entry.entry_category].emplace("epsilon", std::vector<std::string>{ fmt::format("{}", entry.entry_value.epsilon) });
        tracking_entries_[entry.entry_category].emplace("max_iter", std::vector<std::string>{ fmt::format("{}", entry.entry_value.max_iter) });
        tracking_entries_[entry.entry_category].emplace("classification_type", std::vector<std::string>{ fmt::format("{}", entry.entry_value.classification) });
        tracking_entries_[entry.entry_category].emplace("backend", std::vector<std::string>{ fmt::format("{}", entry.entry_value.backend) });
        tracking_entries_[entry.entry_category].emplace("target", std::vector<std::string>{ fmt::format("{}", entry.entry_value.target) });
        tracking_entries_[entry.entry_category].emplace("sycl_kernel_invocation_type", std::vector<std::string>{ fmt::format("{}", entry.entry_value.sycl_kernel_invocation_type) });
        tracking_entries_[entry.entry_category].emplace("sycl_implementation_type", std::vector<std::string>{ fmt::format("{}", entry.entry_value.sycl_implementation_type) });
        tracking_entries_[entry.entry_category].emplace("strings_as_labels", std::vector<std::string>{ fmt::format("{}", entry.entry_value.strings_as_labels) });
        tracking_entries_[entry.entry_category].emplace("real_type", std::vector<std::string>{ std::string{ arithmetic_type_name<real_type>() } });
        tracking_entries_[entry.entry_category].emplace("input_filename", std::vector<std::string>{ fmt::format("\"{}\"", entry.entry_value.input_filename) });
        tracking_entries_[entry.entry_category].emplace("model_filename", std::vector<std::string>{ fmt::format("\"{}\"", entry.entry_value.model_filename) });
    }
}

void performance_tracker::add_tracking_entry(const tracking_entry<cmd::parser_predict> &entry) {
    // check whether entries should currently be tracked
    if (this->is_tracking()) {
        // create category
        tracking_entries_.emplace(entry.entry_category, std::map<std::string, std::vector<std::string>>{});
        // fill category with value
        tracking_entries_[entry.entry_category].emplace("task", std::vector<std::string>{ "predict" });
        tracking_entries_[entry.entry_category].emplace("backend", std::vector<std::string>{ fmt::format("{}", entry.entry_value.backend) });
        tracking_entries_[entry.entry_category].emplace("target", std::vector<std::string>{ fmt::format("{}", entry.entry_value.target) });
        tracking_entries_[entry.entry_category].emplace("sycl_implementation_type", std::vector<std::string>{ fmt::format("{}", entry.entry_value.sycl_implementation_type) });
        tracking_entries_[entry.entry_category].emplace("strings_as_labels", std::vector<std::string>{ fmt::format("{}", entry.entry_value.strings_as_labels) });
        tracking_entries_[entry.entry_category].emplace("real_type", std::vector<std::string>{ std::string{ arithmetic_type_name<real_type>() } });
        tracking_entries_[entry.entry_category].emplace("input_filename", std::vector<std::string>{ fmt::format("\"{}\"", entry.entry_value.input_filename) });
        tracking_entries_[entry.entry_category].emplace("model_filename", std::vector<std::string>{ fmt::format("\"{}\"", entry.entry_value.model_filename) });
        tracking_entries_[entry.entry_category].emplace("predict_filename", std::vector<std::string>{ fmt::format("\"{}\"", entry.entry_value.predict_filename) });
    }
}

void performance_tracker::add_tracking_entry(const tracking_entry<cmd::parser_scale> &entry) {
    // check whether entries should currently be tracked
    if (this->is_tracking()) {
        // create category
        tracking_entries_.emplace(entry.entry_category, std::map<std::string, std::vector<std::string>>{});
        // fill category with value
        tracking_entries_[entry.entry_category].emplace("task", std::vector<std::string>{ "scale" });
        tracking_entries_[entry.entry_category].emplace("lower", std::vector<std::string>{ fmt::format("{}", entry.entry_value.lower) });
        tracking_entries_[entry.entry_category].emplace("upper", std::vector<std::string>{ fmt::format("{}", entry.entry_value.upper) });
        tracking_entries_[entry.entry_category].emplace("format", std::vector<std::string>{ fmt::format("{}", entry.entry_value.format) });
        tracking_entries_[entry.entry_category].emplace("strings_as_labels", std::vector<std::string>{ fmt::format("{}", entry.entry_value.strings_as_labels) });
        tracking_entries_[entry.entry_category].emplace("real_type", std::vector<std::string>{ std::string{ arithmetic_type_name<real_type>() } });
        tracking_entries_[entry.entry_category].emplace("input_filename", std::vector<std::string>{ fmt::format("\"{}\"", entry.entry_value.input_filename) });
        tracking_entries_[entry.entry_category].emplace("scaled_filename", std::vector<std::string>{ fmt::format("\"{}\"", entry.entry_value.scaled_filename) });
        tracking_entries_[entry.entry_category].emplace("save_filename", std::vector<std::string>{ fmt::format("\"{}\"", entry.entry_value.save_filename) });
        tracking_entries_[entry.entry_category].emplace("restore_filename", std::vector<std::string>{ fmt::format("\"{}\"", entry.entry_value.restore_filename) });
    }
}

#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
void performance_tracker::add_hws_entry(const hws::system_hardware_sampler &entry) {
    // check whether entries should currently be tracked
    const std::string entry_category{ "hardware_sampler" };
    if (this->is_tracking()) {
        for (const std::unique_ptr<hws::hardware_sampler> &sampler : entry.samplers()) {
            // get the sample string and append two newlines to each line
            std::string sample_str = sampler->samples_only_as_yaml_string();
            detail::replace_all(sample_str, "\n", "\n    ");

            // generate entry for current sampler
            const std::string entry_string = fmt::format("\n"
                                                         "    sampling_interval: {}\n"
                                                         "    time_points: [{}]\n\n"
                                                         "    {}",
                                                         sampler->sampling_interval(),
                                                         fmt::join(detail::tracking::durations_from_reference_time(sampler->sampling_time_points(), this->reference_time_), ", "),
                                                         sample_str);

            tracking_entries_[entry_category][sampler->device_identification()].push_back(entry_string);
        }
    }
}
#endif

void performance_tracker::add_event(const std::string name) {
    events_.add_event(std::chrono::steady_clock::now(), std::move(name));
}

void performance_tracker::set_reference_time(const std::chrono::steady_clock::time_point time) noexcept {
    reference_time_ = time;
}

std::chrono::steady_clock::time_point performance_tracker::get_reference_time() const noexcept {
    return reference_time_;
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

    //*************************************************************************************************************************************//
    //                                                              meta-data                                                              //
    //*************************************************************************************************************************************//
    // get the current host- and username
#if defined(PLSSVM_UNISTD_AVAILABLE)
    const auto host_name_max = static_cast<std::size_t>(sysconf(_SC_HOST_NAME_MAX));
    std::string hostname(host_name_max, '\0');
    gethostname(hostname.data(), host_name_max);
    const auto login_name_max = static_cast<std::size_t>(sysconf(_SC_LOGIN_NAME_MAX));
    std::string username(login_name_max, '\0');
    getlogin_r(username.data(), login_name_max);
#else
    constexpr std::string_view hostname{ "not available" };
    constexpr std::string_view username{ "not available" };
#endif
    // check whether asserts are enabled
    constexpr bool assert_enabled = PLSSVM_IS_DEFINED(PLSSVM_ENABLE_ASSERTS);
    // check whether LTO has been enabled
    constexpr bool lto_enabled = PLSSVM_IS_DEFINED(PLSSVM_LTO_SUPPORTED);
    // check whether fast-math has been enabled
    constexpr bool fast_math_enabled = PLSSVM_IS_DEFINED(PLSSVM_USE_FAST_MATH);
    // check whether the maximum allocatable memory size should be enforced
    constexpr bool enforce_max_mem_alloc_size = PLSSVM_IS_DEFINED(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE);

    // begin a new YAML document (only with "---" multiple YAML docments in a single file are allowed)
    out << "---\n";

    // output metadata information
    out << fmt::format(
        "meta_data:\n"
        "  date:                              \"{}\"\n"
        "  PLSSVM_TARGET_PLATFORMS:           \"{}\"\n"
        "  commit:                            {}\n"
        "  version:                           {}\n"
        "  hostname:                          {}\n"
        "  user:                              {}\n"
        "  build_type:                        {}\n"
        "  LTO:                               {}\n"
        "  fast-math:                         {}\n"
        "  asserts:                           {}\n"
        "  enforce_max_mem_alloc_size:        {}\n"
        "  THREAD_BLOCK_SIZE:                 {}\n"
        "  FEATURE_BLOCK_SIZE:                {}\n"
        "  INTERNAL_BLOCK_SIZE:               {}\n"
        "  PADDING_SIZE:                      {}\n",
        plssvm::detail::current_date_time(),
        version::detail::target_platforms,
        version::git_metadata::commit_sha1().empty() ? "unknown" : version::git_metadata::commit_sha1(),
        version::version,
        hostname.data(),
        username.data(),
        PLSSVM_BUILD_TYPE,
        lto_enabled,
        fast_math_enabled,
        assert_enabled,
        enforce_max_mem_alloc_size,
        THREAD_BLOCK_SIZE,
        FEATURE_BLOCK_SIZE,
        INTERNAL_BLOCK_SIZE,
        PADDING_SIZE);

#if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    // check whether DPC++ AOT has been enabled
    constexpr bool dpcpp_aot = PLSSVM_IS_DEFINED(PLSSVM_SYCL_BACKEND_DPCPP_ENABLE_AOT);

    out << fmt::format(
        "  DPCPP_backend_type:                {}\n"
        "  DPCPP_amd_gpu_backend_type:        {}\n"
        "  DPCPP_with_aot:                    {}\n",
        PLSSVM_SYCL_BACKEND_DPCPP_BACKEND_TYPE,
        PLSSVM_SYCL_BACKEND_DPCPP_GPU_AMD_BACKEND_TYPE,
        dpcpp_aot);
#endif
#if defined(PLSSVM_SYCL_BACKEND_HAS_ADAPTIVECPP)
    // check whether AdaptiveCpp's new SSCP has been enabled
    constexpr bool adaptivecpp_sscp = PLSSVM_IS_DEFINED(PLSSVM_SYCL_BACKEND_ADAPTIVECPP_USE_GENERIC_SSCP);
    constexpr bool adaptivecpp_accelerated_cpu = PLSSVM_IS_DEFINED(__HIPSYCL_USE_ACCELERATED_CPU__);

    out << fmt::format(
        "  ADAPTIVECPP_with_generic_SSCP:     {}\n"
        "  ADAPTIVECPP_with_accelerated_CPU:  {}\n",
        adaptivecpp_sscp,
        adaptivecpp_accelerated_cpu);
#endif
    out << "\n";

    //*************************************************************************************************************************************//
    //                                                            dependencies                                                             //
    //*************************************************************************************************************************************//
    // cxxopts version
    const std::string cxxopts_version{ fmt::format("{}.{}.{}", CXXOPTS__VERSION_MAJOR, CXXOPTS__VERSION_MINOR, CXXOPTS__VERSION_MINOR) };
    // {fmt} library version
    constexpr int fmt_version_major = FMT_VERSION / 10'000;
    constexpr int fmt_version_minor = FMT_VERSION % 10'000 / 100;
    constexpr int fmt_version_patch = FMT_VERSION % 10;
    const std::string fmt_version{ fmt::format("{}.{}.{}", fmt_version_major, fmt_version_minor, fmt_version_patch) };
    // fast float version
#if defined(PLSSVM_fast_float_VERSION)
    const std::string fast_float_version{ PLSSVM_fast_float_VERSION };
#else
    const std::string fast_float_version{ "unknown" };
#endif
    // igor version
#if defined(PLSSVM_igor_VERSION)
    const std::string igor_version{ PLSSVM_igor_VERSION };
#else
    const std::string igor_version{ "unknown" };
#endif

    // stdpar backend specific versions
    // Boost version
#if defined(PLSSVM_STDPAR_BACKEND_HAS_GNU_TBB)
    const std::string boost_version = fmt::format("{}.{}.{}", BOOST_VERSION / 100'000, BOOST_VERSION / 100 % 1000, BOOST_VERSION % 100);
#else
    const std::string boost_version{ "unknown/unused" };
#endif
    // Intel oneDPL version
#if defined(PLSSVM_STDPAR_BACKEND_HAS_INTEL_LLVM)
    const std::string oneDPL_version = fmt::format("{}.{}.{}", ONEDPL_VERSION_MAJOR, ONEDPL_VERSION_MINOR, ONEDPL_VERSION_PATCH);
#else
    const std::string oneDPL_version{ "unknown/unused" };
#endif
    // Intel TBB version
#if defined(PLSSVM_STDPAR_BACKEND_HAS_ACPP) || defined(PLSSVM_STDPAR_BACKEND_HAS_GNU_TBB)
    const std::string tbb_version = fmt::format("{}.{}", TBB_VERSION_MAJOR, TBB_VERSION_MINOR);
#else
    const std::string tbb_version{ "unknown/unused" };
#endif
#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
    const std::string_view hws_version = hws::version::version;
#else
    const std::string_view hws_version{ "unknown/unused" };
#endif

    out << "dependencies:\n";

    // calculate the number of padding whitespaces for the dependencies category
    std::size_t max_dependency_entry_name_length = 18;  // fast_float_version
    if (detail::contains(tracking_entries_, "dependencies")) {
        for (const auto &[entry_name, entry_value] : tracking_entries_["dependencies"]) {
            max_dependency_entry_name_length = std::max(max_dependency_entry_name_length, entry_name.size());
        }

        // output tracked dependency values
        for (const auto &[entry_name, entry_value] : tracking_entries_["dependencies"]) {
            // check if the current tracking entry contains more than a single value
            if (entry_value.size() == 1) {
                // single value: output it directly
                out << fmt::format("  {}: {:>{}}{}\n", entry_name, "", max_dependency_entry_name_length - entry_name.size(), entry_value.front());
            } else {
                // multiple values: create a YAML array
                out << fmt::format("  {}: {:>{}}[{}]\n", entry_name, "", max_dependency_entry_name_length - entry_name.size(), fmt::join(entry_value, ", "));
            }
        }
    }

    // output the third-party library dependency versions
    out << fmt::format(
        "  cxxopts_version: {}\n"
        "  fmt_version: {}\n"
        "  fast_float_version: {}\n"
        "  igor_version: {}\n"
        "  boost_version: {}\n"
        "  oneDPL_version: {}\n"
        "  tbb_version: {}\n"
        "  hws_version: {}\n\n",
        fmt::format("{:<{}}\"{}\"", "", max_dependency_entry_name_length - 15, cxxopts_version),
        fmt::format("{:<{}}\"{}\"", "", max_dependency_entry_name_length - 11, fmt_version),
        fmt::format("{:<{}}\"{}\"", "", max_dependency_entry_name_length - 18, fast_float_version),
        fmt::format("{:<{}}\"{}\"", "", max_dependency_entry_name_length - 12, igor_version),
        fmt::format("{:<{}}\"{}\"", "", max_dependency_entry_name_length - 13, boost_version),
        fmt::format("{:<{}}\"{}\"", "", max_dependency_entry_name_length - 14, oneDPL_version),
        fmt::format("{:<{}}\"{}\"", "", max_dependency_entry_name_length - 11, tbb_version),
        fmt::format("{:<{}}\"{}\"", "", max_dependency_entry_name_length - 11, hws_version));

    //*************************************************************************************************************************************//
    //                                                          events, if present                                                         //
    //*************************************************************************************************************************************//
    if (!events_.empty()) {
        out << fmt::format("events:\n{}\n\n", events_.generate_yaml_string(reference_time_));
    }

    //*************************************************************************************************************************************//
    //                                                          other statistics                                                           //
    //*************************************************************************************************************************************//
    // output the actual (performance) statistics
    for (const auto &[category, category_entries] : tracking_entries_) {
        // dependencies category handled separately
        if (category == "dependencies") {
            continue;
        }
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

            // check if the current tracking entry contains more than a single value
            if (entries.size() == 1) {
                // single value: output it directly
                out << entries.front();
            } else {
                // multiple values: create a YAML array
                out << fmt::format("[{}]", fmt::join(entries, ", "));
            }
            out << '\n';
        }
        out << '\n';
    }
}

void performance_tracker::pause_tracking() noexcept { is_tracking_ = false; }

void performance_tracker::resume_tracking() noexcept { is_tracking_ = true; }

bool performance_tracker::is_tracking() const noexcept { return is_tracking_; }

const std::map<std::string, std::map<std::string, std::vector<std::string>>> &performance_tracker::get_tracking_entries() const noexcept { return tracking_entries_; }

const events &performance_tracker::get_events() const noexcept { return events_; }

void performance_tracker::clear_tracking_entries() noexcept { tracking_entries_.clear(); }

performance_tracker &global_performance_tracker() {
    static performance_tracker tracker;
    return tracker;
}

}  // namespace plssvm::detail::tracking

#undef PLSSVM_UNISTD_AVAILABLE
