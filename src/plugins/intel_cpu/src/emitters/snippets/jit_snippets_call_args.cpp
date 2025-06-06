// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_snippets_call_args.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "emitters/utils.hpp"
#include "openvino/core/except.hpp"

namespace ov::intel_cpu {

jit_snippets_call_args::~jit_snippets_call_args() {
    delete[] loop_args;
    delete[] external_ptrs;
}

void jit_snippets_call_args::register_loops(const std::vector<loop_args_t>& loops) {
    if (loops.empty()) {
        return;
    }
    const auto num_loops = loops.size();
    OPENVINO_ASSERT(num_loops <= PTRDIFF_MAX, "Requested allocation size { ", num_loops, " } exceeds PTRDIFF_MAX.");
    OPENVINO_ASSERT(loop_args == nullptr, "Loop args are already initialized");
    loop_args = new loop_args_t[static_cast<ptrdiff_t>(num_loops)];
    std::copy(loops.begin(), loops.end(), loop_args);
}

void jit_snippets_call_args::init_external_ptrs(const size_t size) {
    if (size == 0) {
        return;
    }
    OPENVINO_ASSERT(size <= PTRDIFF_MAX, "Requested allocation size { ", size, " } exceeds PTRDIFF_MAX.");
    OPENVINO_ASSERT(external_ptrs == nullptr, "External pointers are already initialized");
    external_ptrs = new const void*[static_cast<ptrdiff_t>(size)];
}

jit_snippets_call_args::loop_args_t::loop_args_t(int64_t work_amount,
                                                 const std::vector<int64_t>& ptr_increments,
                                                 const std::vector<int64_t>& finalization_offsets)
    : m_work_amount(work_amount) {
    OV_CPU_JIT_EMITTER_ASSERT(ptr_increments.size() == finalization_offsets.size(),
                              "Inconsistent sizes of ptr_increments and finalization_offsets");
    m_num_data_ptrs = static_cast<int64_t>(ptr_increments.size());
    init_pointers_and_copy_data(m_num_data_ptrs, ptr_increments.data(), finalization_offsets.data());
}

jit_snippets_call_args::loop_args_t::loop_args_t(const loop_args_t& other)
    : m_work_amount(other.m_work_amount),
      m_num_data_ptrs(other.m_num_data_ptrs) {
    init_pointers_and_copy_data(m_num_data_ptrs, other.m_ptr_increments, other.m_finalization_offsets);
}

jit_snippets_call_args::loop_args_t::~loop_args_t() {
    delete[] m_ptr_increments;
    delete[] m_finalization_offsets;
}

jit_snippets_call_args::loop_args_t& jit_snippets_call_args::loop_args_t::operator=(loop_args_t other) {
    swap(*this, other);
    return *this;
}

void jit_snippets_call_args::loop_args_t::init_pointers_and_copy_data(const int64_t num_elements,
                                                                      const int64_t* ptr_increments,
                                                                      const int64_t* finalization_offsets) {
    const size_t chunk_size = num_elements * sizeof(int64_t);
    OPENVINO_ASSERT(m_ptr_increments == nullptr, "Ptr increments already initialized");
    OPENVINO_ASSERT(m_finalization_offsets == nullptr, "Finalization offsets already initialized");
    m_ptr_increments = new int64_t[num_elements];
    m_finalization_offsets = new int64_t[num_elements];
    std::memcpy(m_ptr_increments, ptr_increments, chunk_size);
    std::memcpy(m_finalization_offsets, finalization_offsets, chunk_size);
}

void swap(jit_snippets_call_args::loop_args_t& first, jit_snippets_call_args::loop_args_t& second) noexcept {
    std::swap(first.m_work_amount, second.m_work_amount);
    std::swap(first.m_num_data_ptrs, second.m_num_data_ptrs);
    std::swap(first.m_ptr_increments, second.m_ptr_increments);
    std::swap(first.m_finalization_offsets, second.m_finalization_offsets);
}

}  // namespace ov::intel_cpu
