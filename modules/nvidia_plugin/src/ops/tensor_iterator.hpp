// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cuda_operation_base.hpp>
#include <cuda/graph.hpp>
#include <kernels/insert.hpp>
#include <kernels/slice.hpp>
#include <openvino/op/tensor_iterator.hpp>

#include "subgraph.hpp"

namespace ov {
namespace nvidia_gpu {

class TensorIteratorOp : public SubGraph {
public:
    using NodeOp = ov::op::v0::TensorIterator;
    TensorIteratorOp(const CreationContext& context,
                     const NodeOp& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    void ExecuteGraph(const InferenceRequestContext& context,
                      Inputs inputTensors,
                      Outputs outputTensors,
                      const Workbuffers& workbuffers);

    bool IsCudaGraphCompatible() const override;

    void Capture(InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

private:
    struct PortMap {
        int64_t start{0};
        int64_t stride{0};
        int64_t part_size{0};
        int64_t end{0};
        int64_t axis{0};
    };

    class SliceLauncher {
    public:
        SliceLauncher(const TensorIteratorOp& ti,
                    //   const CUDA::Stream& stream,
                      const CUDA::DevicePointer<void*> mutableBuffer,
                      const IOperationExec::Inputs& inputTensors,
                      const uint64_t inputIdx,
                      const uint64_t paramIdx);

        inline void operator()(const CUDA::Stream& stream, int64_t iter) const {
            slice_(stream.get(), src_, dst_, start_ + iter * stride_);
        }

        void capture(const CUDA::Stream& stream) {
            slice_node_.emplace(CUDA::CaptureInfo{stream}.addSliceNode(slice_.getParams(src_, dst_, start_)));
        }

        inline void update_capture(const CUDA::GraphExec& exec, int64_t iter) {
            slice_node_->update_params(exec, slice_.getParams(src_, dst_, start_ + iter * stride_));
        }

    private:
        // const CUDA::Stream& stream_;
        const void* src_;
        void* dst_;
        size_t start_;
        int64_t stride_;
        const kernel::Slice& slice_;
        std::optional<CUDA::SliceNode> slice_node_;
    };

    class TransferLauncher {
    public:
        TransferLauncher(const TensorIteratorOp& ti,
                        //  const CUDA::Stream& stream,
                         CUDA::DevicePointer<void*> mutableBuffer,
                         uint64_t resultIdx,
                         uint64_t paramIdx);

        inline void operator()(const CUDA::Stream& stream) const {
            // stream_.transfer(dst_, src_, count_);
            throwIfError(cudaMemcpyAsync(dst_, src_, count_, cudaMemcpyDeviceToDevice, stream.get()));
        }

        void capture(const CUDA::Stream& stream) {
            // TODO: refactor
            transfer_node_.emplace(CUDA::CaptureInfo{stream}.addTransferNode(
                CUDA::DevicePointer<void *>{dst_},
                CUDA::DevicePointer<const void *>{src_},
                count_));
        }

    private:
        // CUDA::DevicePointer<void*> dst_;
        // CUDA::DevicePointer<const void*> src_;
        // const CUDA::Stream& stream_;
        const void* src_;
        void* dst_;
        std::size_t count_;
        std::optional<CUDA::TransferNode> transfer_node_;
    };

    class InsertLauncher {
    public:
        InsertLauncher(const TensorIteratorOp& ti,
                    //    const CUDA::Stream& stream,
                       CUDA::DevicePointer<void*> mutableBuffer,
                       const IOperationExec::Outputs& outputTensors,
                       const std::size_t resultIdx,
                       const std::size_t outputIdx);

        inline void operator()(const CUDA::Stream& stream, int64_t iter) const {
            insert_(stream.get(), src_, dst_, start_ + iter * stride_);
        }

        void capture(const CUDA::Stream& stream) {
            insert_node_.emplace(CUDA::CaptureInfo{stream}.addInsertNode(insert_.getParams(src_, dst_, start_)));
        }

        inline void update_capture(const CUDA::GraphExec& exec, int64_t iter) {
            insert_node_->update_params(exec, insert_.getParams(src_, dst_, start_ + iter * stride_));
        }

    private:
        // const CUDA::Stream& stream_;
        const void* src_;
        void* dst_;
        size_t start_;
        int64_t stride_;
        const kernel::Insert& insert_;
        std::optional<CUDA::InsertNode> insert_node_;
    };

    WorkbufferRequest GetWorkBufferRequest() const override;
    void InitSharedImmutableWorkbuffers(const Buffers& buffers) override;

    void transferParam(const CUDA::Stream& stream,
                       CUDA::DevicePointer<void*> mutableBuffer,
                       const IOperationExec::Inputs& inputTensors,
                       std::int64_t iter,
                       uint64_t inputIdx,
                       uint64_t paramIdx) const;
    void sliceParam(const CUDA::Stream& stream,
                    CUDA::DevicePointer<void*> mutableBuffer,
                    const IOperationExec::Inputs& inputTensors,
                    std::int64_t iter,
                    uint64_t inputIdx,
                    uint64_t paramIdx) const;
    void copyBackEdge(const CUDA::Stream& stream,
                      CUDA::DevicePointer<void*> mutableBuffer,
                      uint64_t resultIdx,
                      uint64_t paramIdx) const;
    void transferResult(const CUDA::Stream& stream,
                        CUDA::DevicePointer<void*> mutableBuffer,
                        const IOperationExec::Outputs& outputTensors,
                        int64_t iter,
                        std::size_t resultIdx,
                        std::size_t outputIdx) const;
    void insertResult(const CUDA::Stream& stream,
                      CUDA::DevicePointer<void*> mutableBuffer,
                      const IOperationExec::Outputs& outputTensors,
                      int64_t iter,
                      std::size_t resultIdx,
                      std::size_t outputIdx) const;

    void updateExecSequence();

    size_t max_threads_per_block_;
    const int64_t num_iterations_;
    std::vector<OperationInfo> inputs_info_;
    std::vector<OperationInfo> outputs_info_;
    std::unordered_map<uint64_t, uint64_t> inputs_parameters_map_;
    std::vector<uint64_t> invariant_inputs_;
    std::unordered_map<uint64_t, PortMap> portmap_inputs_;
    std::unordered_map<uint64_t, kernel::Slice> kernelmap_inputs_;
    std::unordered_map<uint64_t, uint64_t> results_outputs_map_;
    std::unordered_map<uint64_t, std::vector<uint64_t>> iterations_results_map_;
    std::unordered_map<uint64_t, PortMap> portmap_outputs_;
    std::unordered_map<uint64_t, kernel::Insert> kernelmap_outputs_;
    std::unordered_map<uint64_t, uint64_t> results_parameters_map_;

    mutable std::optional<CUDA::Graph> graph_;
    mutable std::optional<CUDA::GraphExec> graph_exec_;
    mutable std::vector<SliceLauncher> slices_;
    mutable std::vector<TransferLauncher> transfers_;
    mutable std::vector<InsertLauncher> inserts_;
};

}  // namespace nvidia_gpu
}  // namespace ov
