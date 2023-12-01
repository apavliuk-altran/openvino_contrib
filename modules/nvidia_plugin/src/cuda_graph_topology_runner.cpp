// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_graph_topology_runner.hpp"

#include "cuda/event.hpp"
#include "ops/tensor_iterator.hpp"

namespace ov {
namespace nvidia_gpu {

CudaGraphTopologyRunner::CudaGraphTopologyRunner(const CreationContext& context, const SubGraph& subgraph, std::size_t treeLevel)
    : orig_subgraph_(subgraph), cuda_graphs_count_{0}, tree_level_{treeLevel} {

    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    for (std::size_t i = 0; i < tree_level_; ++i) {
        std::cout << '\t';
    }
    std::cout << "CudaGraphTopologyRunner::CudaGraphTopologyRunner(): tree_level_ = " << tree_level_ << '\n';
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

    std::vector<SubGraph::ExecSequence> sequences;
    SubGraph::ExecSequence currentSequence;
    const auto& origSequence = orig_subgraph_.getExecSequence();
    const auto totalSize = origSequence.size();
    OPENVINO_ASSERT(totalSize != 0, "ExecSequence size is 0");

    CudaGraphCompatibility lastOpCompatibility = origSequence[0]->GetCudaGraphCompatibility();
    currentSequence.push_back(origSequence[0]);
    for (std::size_t i = 1; i < totalSize; ++i) {
        const auto& op = origSequence[i];
        auto comp = op->GetCudaGraphCompatibility(); 
        if (comp != lastOpCompatibility || comp == CudaGraphCompatibility::SPECIAL) {
            lastOpCompatibility = comp;
            sequences.emplace_back(std::move(currentSequence));
            currentSequence.clear();
        }
        if (comp == CudaGraphCompatibility::SPECIAL) {
            auto sg = std::dynamic_pointer_cast<SubGraph>(op);
            sg->initializeRunner(tree_level_ + 1);
        }
        currentSequence.push_back(op);
    }
    sequences.emplace_back(std::move(currentSequence));

    const auto& model = orig_subgraph_.getModel();
    const auto& memoryManager = orig_subgraph_.memoryManager();
    for (const auto& sequence : sequences) {
        // subgraphs_.emplace_back(context, model, std::move(sequence), memoryManager);
        subgraphs_.emplace_back(context, model, sequence, memoryManager);
        if (subgraphs_.back().GetCudaGraphCompatibility() != CudaGraphCompatibility::NONE) {
            ++cuda_graphs_count_;
        }
    }
}

CudaGraphTopologyRunner::CudaGraphTopologyRunner(const CreationContext& context,
                                                 const std::shared_ptr<const ov::Model>& model,
                                                 std::size_t treeLevel)
    : CudaGraphTopologyRunner(context, {context, model}, treeLevel) {}
//     : orig_subgraph_{context, model}, cuda_graphs_count_{0}, tree_level_{treeLevel} {
// }

CudaGraphTopologyRunner::CudaGraphTopologyRunner(const CreationContext& context,
                                                 const std::shared_ptr<const ov::Model>& model,
                                                 const SubGraph::ExecSequence& sequence,
                                                 const std::shared_ptr<MemoryManager>& memoryManager,
                                                 std::size_t treeLevel)
    : CudaGraphTopologyRunner(context, {context, model, sequence, memoryManager}, treeLevel) {}

void CudaGraphTopologyRunner::Run(InferenceRequestContext& context, const Workbuffers& workbuffers) const {
// void CudaGraphTopologyRunner::Run(InferenceRequestContext& context, const Workbuffers& workbuffers, ICudaGraphInfo& graphContext) const {
    const auto& stream = context.getThreadContext().stream();
    // auto& graphContext = context.getCudaGraphContext();
    // auto& graphContext = context.getCudaGraphContext().get_current_graph();
    auto& graphContext = context.getCurrentCudaGraphInfo();
    std::size_t graphIndex = 0;
    for (auto& subgraph : subgraphs_) {
        auto compatibility = subgraph.GetCudaGraphCompatibility();
        if (compatibility == CudaGraphCompatibility::FULL) {
            graphContext.select_current_graph(graphIndex);
            // graphContext.get_current_graph().launch(stream);
            graphContext.launch(stream);
            graphIndex++;
        } else if (compatibility == CudaGraphCompatibility::SPECIAL) {
            graphContext.select_current_graph(graphIndex);
            context.setCurrentCudaGraphInfo(graphContext.get_current_graph());
            subgraph.ExecuteGraph(context, {}, {}, workbuffers);
            graphIndex++;
        // } else if (compatibility == CudaGraphCompatibility::NESTED) {
        //     graphContext.select_current_graph(graphIndex);
        //     subgraph.getRunner().Run(context, workbuffers);
        //     graphIndex++;
        } else {
            subgraph.Execute(context, {}, {}, workbuffers);
        }
    }
}

void CudaGraphTopologyRunner::Run(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const {
    Workbuffers workbuffers{};
    workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
    context.setCurrentCudaGraphInfo(context.getCudaGraphContext());
    // Run(context, workbuffers, context.getCudaGraphContext());
    Run(context, workbuffers);
}

void CudaGraphTopologyRunner::Capture(InferenceRequestContext& context, const Workbuffers& workbuffers) const {
// void CudaGraphTopologyRunner::Capture(InferenceRequestContext& context, const Workbuffers& workbuffers, ICudaGraphInfo& graphContext) const {
    const auto& stream = context.getThreadContext().stream();
    // auto& graphInfo = context.getCudaGraphContext().get_current_graph();
    auto& graphContext = context.getCurrentCudaGraphInfo();
    graphContext.reset();
    for (const auto& subgraph : subgraphs_) {
        auto compatibility = subgraph.GetCudaGraphCompatibility();
        if (compatibility == CudaGraphCompatibility::FULL) {
            graphContext.add(CudaGraphInfo::create());
            CUDA::GraphCapture capture{stream};
            {
                auto scope = capture.getScope();
                subgraph.Capture(context, {}, {}, workbuffers);
            }
            // const auto& graph = capture.getGraph();
            // graphInfo.set_current_graph(graph);
            graphContext.set_current_graph(capture.getGraph());
        } else if (compatibility == CudaGraphCompatibility::SPECIAL) {
            // graphInfo.add(CudaGraphInfo::create());
            // graphContext.add(CudaGraphContext::create());
            auto& currentGraphContext = graphContext.add(CudaGraphContext::create());
            context.setCurrentCudaGraphInfo(currentGraphContext);
            subgraph.Capture(context, {}, {}, workbuffers);
        // } else if (compatibility == CudaGraphCompatibility::NESTED) {
        //     graphInfo.add(CudaGraphContext::create());
        //     // subgraph.Capture(context, {}, {}, workbuffers);
        //     subgraph.getRunner().Capture(context, workbuffers);
        }
    }
}

void CudaGraphTopologyRunner::Capture(InferenceRequestContext& context,
                                      const DeviceMemBlock& memoryBlock) const {
    Workbuffers workbuffers{};
    workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
    // auto& graphInfo = context.getCudaGraphContext();
    // Capture(graphInfo, workbuffers);
    context.setCurrentCudaGraphInfo(context.getCudaGraphContext());
    // Capture(context, workbuffers, context.getCudaGraphContext());
    Capture(context, workbuffers);
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
