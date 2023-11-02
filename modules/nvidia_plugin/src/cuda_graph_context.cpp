// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_graph_context.hpp"

namespace ov {
namespace nvidia_gpu {

void CudaGraphContext::reset() {
    graphs_.clear();
    currentGraphIndex_ = 0;
}

void CudaGraphContext::start_next_graph_addition() {
    currentGraphIndex_ = graphs_.size();
    graphs_.emplace_back();
}

void CudaGraphContext::start_ti_graph_addition(const std::string& ti_op_name) {
    // OPENVINO_ASSERT(!ti_graphs_.is_initialized(), "Only one TI graph supported");
    // ti_graphs_.emplace();
    ti_graphs_[ti_op_name] = {};
    // OPENVINO_ASSERT(ti_graphs_.get_transfers_count() == 0 &&
    //                 ti_graphs_.get_slices_count() == 0 &&
    //                 ti_graphs_.get_inserts_count() == 0,
    //                 "ti_graphs_ hasn't been reset properly");
}

void CudaGraphContext::add_parameter(const std::string& tensorName,
                                     const CUDA::Stream& stream,
                                     CUDA::DevicePointer<void*> dst,
                                     const void* src,
                                     std::size_t size) {
    OPENVINO_ASSERT(currentGraphIndex_ < graphs_.size(), "Graph index/vector size incosistency");
    graphs_[currentGraphIndex_].add_parameter(tensorName, stream, dst, src, size);
}

void CudaGraphContext::add_result(const std::string& tensorName,
                                  const CUDA::Stream& stream,
                                  void* dst,
                                  CUDA::DevicePointer<const void*> src,
                                  std::size_t size) {
    OPENVINO_ASSERT(currentGraphIndex_ < graphs_.size(), "Graph index/vector size incosistency");
    graphs_[currentGraphIndex_].add_result(tensorName, stream, dst, src, size);
}

void CudaGraphContext::add_transfer(const std::string& ti_op_name,
                                    const CUDA::Stream& stream,
                                    CUDA::DevicePointer<void*> dst,
                                    CUDA::DevicePointer<const void*> src,
                                    std::size_t size) {
    // OPENVINO_ASSERT(ti_graphs_.is_initialized(), "TI graph not initialized");
    ti_graphs_.at(ti_op_name).add_transfer(stream, dst, src, size);
}

void CudaGraphContext::add_slice(const std::string& ti_op_name,
                                 const CUDA::Stream& stream,
                                 std::unique_ptr<ov::nvidia_gpu::kernel::Slice::Params> sliceParams) {
    // OPENVINO_ASSERT(ti_graphs_.is_initialized(), "TI graph not initialized");
    ti_graphs_.at(ti_op_name).add_slice(stream, std::move(sliceParams));
}

void CudaGraphContext::add_insert(const std::string& ti_op_name,
                                  const CUDA::Stream& stream,
                                  std::unique_ptr<ov::nvidia_gpu::kernel::Insert::Params> insertParams) {
    // OPENVINO_ASSERT(ti_graphs_.is_initialized(), "TI graph not initialized");
    ti_graphs_.at(ti_op_name).add_insert(stream, std::move(insertParams));
}

void CudaGraphContext::add_graph(const CUDA::Graph& graph) {
    OPENVINO_ASSERT(currentGraphIndex_ < graphs_.size(), "Graph index/vector size incosistency");
    graphs_[currentGraphIndex_].set_graph(graph);
}

void CudaGraphContext::add_ti_graph(const std::string& ti_op_name, const CUDA::Graph& graph) {
    // OPENVINO_ASSERT(ti_graphs_.is_initialized(), "TI graph not initialized");
    ti_graphs_.at(ti_op_name).set_graph(graph);
}

bool CudaGraphContext::is_initialized() const {
    const auto size = graphs_.size();
    return size != 0 && graphs_[size - 1].is_initialized();
}

void CudaGraphContext::update_capture(const TensorMappingContext& context) {
    for (currentGraphIndex_ = 0; currentGraphIndex_ < graphs_.size(); ++currentGraphIndex_) {
        graphs_[currentGraphIndex_].update_capture(context);
    }
}

void CudaGraphContext::update_slice(const std::string& ti_op_name,
                                    std::size_t index,
                                    std::unique_ptr<ov::nvidia_gpu::kernel::Slice::Params> sliceParams) const {
    OPENVINO_ASSERT(ti_graphs_.at(ti_op_name).is_initialized(), "TI graph not initialized");
    ti_graphs_.at(ti_op_name).update_slice(index, std::move(sliceParams));
}

void CudaGraphContext::update_insert(const std::string& ti_op_name,
                                     std::size_t index,
                                     std::unique_ptr<ov::nvidia_gpu::kernel::Insert::Params> insertParams) const {
    OPENVINO_ASSERT(ti_graphs_.at(ti_op_name).is_initialized(), "TI graph not initialized");
    ti_graphs_.at(ti_op_name).update_insert(index, std::move(insertParams));
}

void CudaGraphContext::launch(std::size_t index, const CUDA::Stream& stream) const {
    currentGraphIndex_ = index;
    OPENVINO_ASSERT(currentGraphIndex_ < graphs_.size(), "Graph index/vector size incosistency");
    graphs_[currentGraphIndex_].launch(stream);
}

void CudaGraphContext::launch_ti_graph(const std::string& ti_op_name, const CUDA::Stream& stream) const {
    OPENVINO_ASSERT(ti_graphs_.at(ti_op_name).is_initialized(), "TI graph not initialized");
    ti_graphs_.at(ti_op_name).launch(stream);
}

std::size_t CudaGraphContext::get_params_count() const {
    std::size_t res = 0;
    for (const auto& graph : graphs_) {
        res += graph.get_params_count();
    }
    return res;
}

std::size_t CudaGraphContext::get_results_count() const {
    std::size_t res = 0;
    for (const auto& graph : graphs_) {
        res += graph.get_results_count();
    }
    return res;
}

std::size_t CudaGraphContext::get_transfers_count(const std::string& ti_op_name) const {
        return ti_graphs_.at(ti_op_name).get_transfers_count();
}


std::size_t CudaGraphContext::get_slices_count(const std::string& ti_op_name) const {
    return ti_graphs_.at(ti_op_name).get_slices_count();
}

std::size_t CudaGraphContext::get_inserts_count(const std::string& ti_op_name) const {
    return ti_graphs_.at(ti_op_name).get_inserts_count();
}

std::size_t CudaGraphContext::get_graphs_count() const {
    return graphs_.size();
}

void CudaGraphContext::CudaGraphInfo::add_parameter(const std::string& tensorName,
                                                    const CUDA::Stream& stream,
                                                    CUDA::DevicePointer<void*> dst,
                                                    const void* src,
                                                    std::size_t size) {
    CUDA::CaptureInfo captureInfo{stream};
    parameterNodes_.emplace(tensorName, captureInfo.addUploadNode(dst, src, size));
}

void CudaGraphContext::CudaGraphInfo::add_result(const std::string& tensorName,
                                                 const CUDA::Stream& stream,
                                                 void* dst,
                                                 CUDA::DevicePointer<const void*> src,
                                                 std::size_t size) {
    CUDA::CaptureInfo captureInfo{stream};
    resultNodes_.emplace(tensorName, captureInfo.addDownloadNode(dst, src, size));
}

void CudaGraphContext::CudaGraphInfo::add_transfer(const CUDA::Stream& stream,
                                                   CUDA::DevicePointer<void*> dst,
                                                   CUDA::DevicePointer<const void*> src,
                                                   std::size_t size) {
    CUDA::CaptureInfo captureInfo{stream};
    transferNodes_.emplace_back(captureInfo.addTransferNode(dst, src, size));
}

void CudaGraphContext::CudaGraphInfo::add_slice(const CUDA::Stream& stream,
                                                std::unique_ptr<ov::nvidia_gpu::kernel::Slice::Params> sliceParams) {
    CUDA::CaptureInfo captureInfo{stream};
    sliceNodes_.emplace_back(captureInfo.addSliceNode(std::move(sliceParams)));
}

void CudaGraphContext::CudaGraphInfo::add_insert(const CUDA::Stream& stream,
                                                 std::unique_ptr<ov::nvidia_gpu::kernel::Insert::Params> insertParams) {
    CUDA::CaptureInfo captureInfo{stream};
    insertNodes_.emplace_back(captureInfo.addInsertNode(std::move(insertParams)));
}

void CudaGraphContext::CudaGraphInfo::set_graph(const CUDA::Graph& graph) {
    graph_.emplace(graph);
    graphExec_.emplace(graph);
}

bool CudaGraphContext::CudaGraphInfo::is_initialized() const { return graph_.has_value() && graphExec_.has_value(); }

void CudaGraphContext::CudaGraphInfo::update_capture(const TensorMappingContext& context) {
    for (auto&& [tensorName, node] : parameterNodes_) {
        node.update_src(graphExec_.value(), (context.get_input_tensor(tensorName)->data()));
    }
    for (auto&& [tensorName, node] : resultNodes_) {
        node.update_dst(graphExec_.value(), context.get_output_tensor(tensorName)->data());
    }
}

void CudaGraphContext::CudaGraphInfo::update_slice(std::size_t index,
                                                   std::unique_ptr<ov::nvidia_gpu::kernel::Slice::Params> sliceParams) {
    sliceNodes_[index].update_params(graphExec_.value(), std::move(sliceParams));
}

void CudaGraphContext::CudaGraphInfo::update_insert(
    std::size_t index, std::unique_ptr<ov::nvidia_gpu::kernel::Insert::Params> insertParams) {
    insertNodes_[index].update_params(graphExec_.value(), std::move(insertParams));
}

void CudaGraphContext::CudaGraphInfo::launch(const CUDA::Stream& stream) const { graphExec_.value().launch(stream); }

std::size_t CudaGraphContext::CudaGraphInfo::get_params_count() const { return parameterNodes_.size(); }

std::size_t CudaGraphContext::CudaGraphInfo::get_results_count() const { return resultNodes_.size(); }

std::size_t CudaGraphContext::CudaGraphInfo::get_transfers_count() const { return transferNodes_.size(); }

std::size_t CudaGraphContext::CudaGraphInfo::get_slices_count() const { return sliceNodes_.size(); }

std::size_t CudaGraphContext::CudaGraphInfo::get_inserts_count() const { return insertNodes_.size(); }

bool operator==(const CudaGraphContext::CudaGraphInfo& lhs, const CudaGraphContext::CudaGraphInfo& rhs) {
    return lhs.graph_ == rhs.graph_ && lhs.graphExec_ == rhs.graphExec_ && lhs.parameterNodes_ == rhs.parameterNodes_ &&
           lhs.resultNodes_ == rhs.resultNodes_;
}

bool operator!=(const CudaGraphContext::CudaGraphInfo& lhs, const CudaGraphContext::CudaGraphInfo& rhs) {
    return !(lhs == rhs);
}

bool operator==(const CudaGraphContext& lhs, const CudaGraphContext& rhs) { return lhs.graphs_ == rhs.graphs_; }

bool operator!=(const CudaGraphContext& lhs, const CudaGraphContext& rhs) { return !(lhs == rhs); }

}  // namespace nvidia_gpu
}  // namespace ov
