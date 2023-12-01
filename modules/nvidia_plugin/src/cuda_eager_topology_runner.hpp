// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ops/subgraph.hpp>

#include "cuda_itopology_runner.hpp"

namespace ov {
namespace nvidia_gpu {

class EagerTopologyRunner final : public SubGraph, public ITopologyRunner {
public:
    EagerTopologyRunner(const CreationContext& context, const std::shared_ptr<const ov::Model>& model) : SubGraph(context, model) {}

    ~EagerTopologyRunner() override = default;

    void Run(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const override {
        Workbuffers workbuffers{};
        workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
        SubGraph::Execute(context, {}, {}, workbuffers);
    }

    const SubGraph& GetSubGraph() const override {
        return *this;
    }

// private:
    void UpdateContext(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const override {};
    void Run(InferenceRequestContext& context, const Workbuffers& workbuffers) const override {};
    // void Run(InferenceRequestContext& context, const Workbuffers& workbuffers, ICudaGraphInfo& graphContext) const override {}
    //TODO: Use signature from SubGraph??
    void Capture(InferenceRequestContext& context, const Workbuffers& workbuffers) const override {};
    // void Capture(InferenceRequestContext& context, const Workbuffers& workbuffers, ICudaGraphInfo& graphContext) const override {}
};

}  // namespace nvidia_gpu
}  // namespace ov
