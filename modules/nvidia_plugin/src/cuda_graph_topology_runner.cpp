// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_graph_topology_runner.hpp"

#include "cuda/event.hpp"
#include "ops/tensor_iterator.hpp"

namespace ov {
namespace nvidia_gpu {

namespace {

std::shared_ptr<TensorIteratorOp> getTI(const SubGraph& sg) {
    auto& seq = sg.getExecSequence();
    if (seq.size() != 1) {
        return nullptr;
    }
    return std::dynamic_pointer_cast<TensorIteratorOp>(seq[0]);
}

} // namespace


CudaGraphTopologyRunner::CudaGraphTopologyRunner(const CreationContext& context,
                                                 const std::shared_ptr<const ov::Model>& model)
    : orig_subgraph_{context, model},
      cuda_graphs_count_{0} {
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    std::cout << "CudaGraphTopologyRunner::CudaGraphTopologyRunner()\n";
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";

    std::vector<SubGraph::ExecSequence> sequences;
    SubGraph::ExecSequence currentSequence;
    const auto& origSequence = orig_subgraph_.getExecSequence();
    const auto totalSize = origSequence.size();
    OPENVINO_ASSERT(totalSize != 0, "ExecSequence size is 0");

    bool isLastOpCompatible = origSequence[0]->IsCudaGraphCompatible();
    currentSequence.push_back(origSequence[0]);
    for (size_t i = 1; i < totalSize; ++i) {
        const auto& op = origSequence[i];
        if (std::dynamic_pointer_cast<const TensorIteratorOp>(op)) {
            isLastOpCompatible = !isLastOpCompatible;
            sequences.emplace_back(std::move(currentSequence));
            currentSequence.clear();
        } else if (op->IsCudaGraphCompatible() != isLastOpCompatible) {
            isLastOpCompatible = !isLastOpCompatible;
            sequences.emplace_back(std::move(currentSequence));
            currentSequence.clear();
        }
        currentSequence.push_back(op);
    }
    sequences.emplace_back(std::move(currentSequence));

    const auto& memoryManager = orig_subgraph_.memoryManager();
    for (auto&& sequence : sequences) {
        subgraphs_.emplace_back(context, model, std::move(sequence), memoryManager);
        if (subgraphs_[subgraphs_.size() - 1].IsCudaGraphCompatible()) {
            ++cuda_graphs_count_;
        }
    }
}

void CudaGraphTopologyRunner::Run(const InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const {
    const auto& stream = context.getThreadContext().stream();
    std::size_t graphIndex = 0;
    for (auto& subgraph : subgraphs_) {
        if (auto ti = getTI(subgraph)) {
            Workbuffers workbuffers{};
            workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
            const auto& mutableBuffer = workbuffers.mutable_buffers.at(0);
            const auto& memoryManager = *subgraph.memoryManager();
            const auto& inputTensors = memoryManager.inputTensorPointers(*ti, mutableBuffer);
            const auto& outputTensors = memoryManager.outputTensorPointers(*ti, mutableBuffer);
            const auto& workBuffers = memoryManager.workBuffers(*ti, mutableBuffer);
            // TODO: avoid const_cast
            ti->ExecuteGraph(const_cast<InferenceRequestContext&>(context), inputTensors, outputTensors, workBuffers);
        } else if (subgraph.IsCudaGraphCompatible()) {
            context.getCudaGraphContext().launch(graphIndex, stream);
            graphIndex++;
        } else {
            Workbuffers workbuffers{};
            workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
            subgraph.Execute(context, {}, {}, workbuffers);
        }
    }
}

void CudaGraphTopologyRunner::Capture(InferenceRequestContext& context,
                                      const DeviceMemBlock& memoryBlock) const {
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    const auto& stream = context.getThreadContext().stream();
    auto& graphContext = context.getCudaGraphContext();

    graphContext.reset();
    for (const auto& subgraph : subgraphs_) {
        if (getTI(subgraph)) {
            Workbuffers workbuffers{};
            workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
            subgraph.Capture(context, {}, {}, workbuffers);
        } else if (subgraph.IsCudaGraphCompatible()) {
            graphContext.start_next_graph_addition();
            CUDA::GraphCapture capture{stream};
            {
                auto scope = capture.getScope();
                Workbuffers workbuffers{};
                workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
                subgraph.Capture(context, {}, {}, workbuffers);
            }
            const auto& graph = capture.getGraph();
            graphContext.add_graph(graph);
        }
    }
    // OPENVINO_ASSERT(graphContext.get_graphs_count() == GetCudaGraphsCount(),
    //                 "CudaGraphTopologyRunner/CudaGraphContext graphs count mismatch");

    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    // std::cout << "CudaGraphTopologyRunner::Capture(): " <<
    //              std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms\n";
    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
}

const SubGraph& CudaGraphTopologyRunner::GetSubGraph() const {
    return orig_subgraph_;
}

std::size_t CudaGraphTopologyRunner::GetCudaGraphsCount() const { return cuda_graphs_count_; }

void CudaGraphTopologyRunner::UpdateContext(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const {
    if (context.getCudaGraphContext().is_initialized()) {
        UpdateCapture(context);
    } else {
        Capture(context, memoryBlock);
    }
}

void CudaGraphTopologyRunner::UpdateCapture(InferenceRequestContext& context) const {
    context.getCudaGraphContext().update_capture(context.getTensorMappingContext());
}

}  // namespace nvidia_gpu
}  // namespace ov
