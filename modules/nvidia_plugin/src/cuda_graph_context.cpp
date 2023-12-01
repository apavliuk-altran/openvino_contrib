// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_graph_context.hpp"

namespace ov {
namespace nvidia_gpu {

void CudaGraphInfo::reset() {
    graph_.reset();
    graphExec_.reset();
    parameterNodes_.clear();
    resultNodes_.clear();
    transferNodes_.clear();
    kernelNodes_.clear();
}

void CudaGraphInfo::add_parameter(const std::string& tensorName,
                                  const CUDA::Stream& stream,
                                  CUDA::DevicePointer<void*> dst,
                                  const void* src,
                                  std::size_t size) {
    CUDA::CaptureInfo captureInfo{stream};
    parameterNodes_.emplace(tensorName, captureInfo.addUploadNode(dst, src, size));
}

void CudaGraphInfo::add_result(const std::string& tensorName,
                               const CUDA::Stream& stream,
                               void* dst,
                               CUDA::DevicePointer<const void*> src,
                               std::size_t size) {
    CUDA::CaptureInfo captureInfo{stream};
    resultNodes_.emplace(tensorName, captureInfo.addDownloadNode(dst, src, size));
}

void CudaGraphInfo::add_transfer(const CUDA::Stream& stream,
                                 CUDA::DevicePointer<void*> dst,
                                 CUDA::DevicePointer<const void*> src,
                                 std::size_t size) {
    CUDA::CaptureInfo captureInfo{stream};
    transferNodes_.emplace_back(captureInfo.addTransferNode(dst, src, size));
}

bool CudaGraphInfo::is_initialized() const { return graph_.has_value() && graphExec_.has_value(); }

void CudaGraphInfo::update_capture(const TensorMappingContext& context) {
    for (auto&& [tensorName, node] : parameterNodes_) {
        node.update_src(graphExec_.value(), (context.get_input_tensor(tensorName)->data()));
    }
    for (auto&& [tensorName, node] : resultNodes_) {
        node.update_dst(graphExec_.value(), context.get_output_tensor(tensorName)->data());
    }
}

std::size_t CudaGraphInfo::get_graphs_count() const {
    return is_initialized() ? 1 : 0;
}

// void CudaGraphInfo::set_graph(const CUDA::Graph& graph) {
//     graph_.emplace(graph);
//     graphExec_.emplace(graph);
// }

void CudaGraphInfo::launch(const CUDA::Stream& stream) const { graphExec_.value().launch(stream); }

// bool operator==(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs) {
//     return lhs.graph_ == rhs.graph_ && lhs.graphExec_ == rhs.graphExec_ && lhs.parameterNodes_ == rhs.parameterNodes_ &&
//            lhs.resultNodes_ == rhs.resultNodes_ && lhs.transferNodes_ == rhs.transferNodes_ &&
//            lhs.kernelNodes_ == rhs.kernelNodes_;
// }

// bool operator!=(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs) { return !(lhs == rhs); }

// CudaGraphContext::CudaGraphContext(const CudaGraphContext& rhs) {
//     for (const auto& graph : rhs.graphs_) {
//         graphs_.emplace_back(std::make_shared<>graph);
//     }
// }

// CudaGraphContext::operator=(const CudaGraphContext & rhs) {}

void CudaGraphContext::reset() {
    graphs_.clear();
    currentGraphIndex_ = 0;
}

void CudaGraphContext::add_parameter(const std::string& tensorName,
                                     const CUDA::Stream& stream,
                                     CUDA::DevicePointer<void*> dst,
                                     const void* src,
                                     std::size_t size) {
    OPENVINO_ASSERT(currentGraphIndex_ < graphs_.size(), "Graph index/vector size incosistency");
    graphs_[currentGraphIndex_]->add_parameter(tensorName, stream, dst, src, size);
}

void CudaGraphContext::add_result(const std::string& tensorName,
                                  const CUDA::Stream& stream,
                                  void* dst,
                                  CUDA::DevicePointer<const void*> src,
                                  std::size_t size) {
    OPENVINO_ASSERT(currentGraphIndex_ < graphs_.size(), "Graph index/vector size incosistency");
    graphs_[currentGraphIndex_]->add_result(tensorName, stream, dst, src, size);
}

void CudaGraphContext::add_transfer(const CUDA::Stream& stream,
                                    CUDA::DevicePointer<void*> dst,
                                    CUDA::DevicePointer<const void*> src,
                                    std::size_t size) {
    graphs_[currentGraphIndex_]->add_transfer(stream, dst, src, size);
}

void CudaGraphContext::set_current_graph(const CUDA::Graph& graph) {
    OPENVINO_ASSERT(currentGraphIndex_ < graphs_.size(), "Graph index/vector size incosistency");
    // graphs_[currentGraphIndex_]->set_graph(graph);
    graphs_[currentGraphIndex_]->set_current_graph(graph);
}

bool CudaGraphContext::is_initialized() const {
    const auto size = graphs_.size();
    return size != 0 && graphs_[size - 1]->is_initialized();
}

void CudaGraphContext::update_capture(const TensorMappingContext& context) {
    for (currentGraphIndex_ = 0; currentGraphIndex_ < graphs_.size(); ++currentGraphIndex_) {
        graphs_[currentGraphIndex_]->update_capture(context);
    }
}

// void CudaGraphContext::add(std::shared_ptr<ICudaGraphInfo> ptr) {
ICudaGraphInfo& CudaGraphContext::add(std::shared_ptr<ICudaGraphInfo> ptr) {
// void CudaGraphContext::add(std::unique_ptr<ICudaGraphInfo> ptr) {
    currentGraphIndex_ = graphs_.size();
    // if (type == GraphInfoType::INFO) {
    //     graphs_.emplace_back(std::make_unique<CudaGraphInfo>());
    // } else if (type == GraphInfoType::CONTEXT) {
    //     graphs_.emplace_back(std::make_unique<CudaGraphContext>());
    // }
    // graphs_.emplace_back(std::make_shared<CudaGraphInfo>());
    graphs_.emplace_back(ptr);
    return *graphs_.back();
}

// TODO: rename?
// const ICudaGraphInfo& CudaGraphContext::get_current_graph() const {
//     const auto* graphInfoPtr = graphs_[currentGraphIndex_].get();
//     while (graphInfoPtr->is_nested()) {
//         graphInfoPtr = &(graphInfoPtr->get_current_graph());
//     }
//     return *graphInfoPtr;
// }

// TODO: rename?
// ICudaGraphInfo& CudaGraphContext::get_current_graph() {
//     if (graphs_.size() == 0) {
//         return *this;
//     }
//     auto* graphInfoPtr = graphs_[currentGraphIndex_].get();
//     while (graphInfoPtr->is_nested()) {
//         graphInfoPtr = &(graphInfoPtr->get_current_graph());
//     }
//     return *graphInfoPtr;
// }
// ICudaGraphInfo& CudaGraphContext::get_current_graph_info() { return *graphs_[currentGraphIndex_]; }
ICudaGraphInfo& CudaGraphContext::get_current_graph() { return *graphs_[currentGraphIndex_]; }

void CudaGraphContext::select_current_graph(std::size_t index) {
    OPENVINO_ASSERT(index < graphs_.size(), "Graph index/vector size incosistency");
    currentGraphIndex_ = index;
}

std::size_t CudaGraphContext::get_params_count() const {
    // TODO: use STL algo
    std::size_t res = 0;
    for (const auto& graph : graphs_) {
        res += graph->get_params_count();
    }
    return res;
}

std::size_t CudaGraphContext::get_results_count() const {
    // TODO: use STL algo
    std::size_t res = 0;
    for (const auto& graph : graphs_) {
        res += graph->get_results_count();
    }
    return res;
}

std::size_t CudaGraphContext::get_transfers_count() const {
    // TODO: use STL algo
    std::size_t res = 0;
    for (const auto& graph : graphs_) {
        res += graph->get_transfers_count();
    }
    return res;
}

std::size_t CudaGraphContext::get_kernels_count() const {
    // TODO: use STL algo
    std::size_t res = 0;
    for (const auto& graph : graphs_) {
        res += graph->get_kernels_count();
    }
    return res;
}

std::size_t CudaGraphContext::get_graphs_count() const { return graphs_.size(); }

// void CudaGraphContext::set_graph(const CUDA::Graph& graph) {
//     graphs_[currentGraphIndex_]->set_graph(graph);
// }

void CudaGraphContext::launch(const CUDA::Stream& stream) const {
    graphs_[currentGraphIndex_]->launch(stream);

    CudaGraphInfo info;
    CudaGraphContext context;
}

// bool operator==(const CudaGraphContext& lhs, const CudaGraphContext& rhs) { return lhs.graphs_ == rhs.graphs_; }

// bool operator!=(const CudaGraphContext& lhs, const CudaGraphContext& rhs) { return !(lhs == rhs); }

}  // namespace nvidia_gpu
}  // namespace ov
