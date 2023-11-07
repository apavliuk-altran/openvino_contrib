// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_iterator.hpp"

#include <cstdint>
#include <cuda_op_buffers_extractor.hpp>
#include <cuda_iexecution_delegator.hpp>
#include <kernels/details/cuda_type_traits.hpp>
#include <kernels/details/tensor_helpers.hpp>
#include <kernels/insert.hpp>
#include <kernels/slice.hpp>

#include "converters.hpp"
#include "cuda_operation_registry.hpp"
#include "parameter.hpp"
#include "result.hpp"

namespace ov {
namespace nvidia_gpu {

TensorIteratorOp::TensorIteratorOp(const CreationContext& context,
                                   const NodeOp& op,
                                   IndexCollection&& inputIds,
                                   IndexCollection&& outputIds)
    : SubGraph(context, op, std::move(inputIds), std::move(outputIds)), num_iterations_{op.get_num_iterations()} {
    // Set trip count, initial execution condition, num iteration primitives
    // they should be mutable_data to prevent from being optimized out
    if (num_iterations_ < 0) {
        throw std::runtime_error("tensor iterator's num_iteration cannot be negative");
    }

    inputs_info_.reserve(op.inputs().size());
    for (auto& input : op.inputs()) {
        inputs_info_.emplace_back(getTensorByteSize(input), input.get_element_type(), input.get_shape());
    }

    outputs_info_.reserve(op.outputs().size());
    for (auto& output : op.outputs()) {
        outputs_info_.emplace_back(getTensorByteSize(output), output.get_element_type(), output.get_shape());
    }

    // Setup input_primitive_maps/ output_primitive_maps and back_edges
    const auto& loop_input_descs = op.get_input_descriptions();
    const auto& loop_output_descs = op.get_output_descriptions();

    // Set input mapping & back edges
    for (const auto& loop_input_desc : loop_input_descs) {
        inputs_parameters_map_[loop_input_desc->m_input_index] = loop_input_desc->m_body_parameter_index;

        // Set invariant input
        if (const auto& invariantInput =
                std::dynamic_pointer_cast<ov::op::util::SubGraphOp::InvariantInputDescription>(loop_input_desc)) {
            invariant_inputs_.push_back(invariantInput->m_input_index);
        }

        // Set input mapping
        if (const auto& sliceInfo = std::dynamic_pointer_cast<NodeOp::SliceInputDescription>(loop_input_desc)) {
            // sliced input
            portmap_inputs_[loop_input_desc->m_input_index] = PortMap{
                sliceInfo->m_start,
                sliceInfo->m_stride,
                sliceInfo->m_part_size,
                sliceInfo->m_end,
                sliceInfo->m_axis,
            };
        }

        // set back edges
        if (const auto& mergedInput = std::dynamic_pointer_cast<NodeOp::MergedInputDescription>(loop_input_desc)) {
            // backedge
            results_parameters_map_[mergedInput->m_body_value_index] = mergedInput->m_body_parameter_index;
        }
    }

    // Set output mapping
    for (const auto& loop_output_desc : loop_output_descs) {
        results_outputs_map_[loop_output_desc->m_body_value_index] = loop_output_desc->m_output_index;

        if (const auto& concatOutput = std::dynamic_pointer_cast<NodeOp::ConcatOutputDescription>(loop_output_desc)) {
            // concat output
            portmap_outputs_[loop_output_desc->m_output_index] = PortMap{
                concatOutput->m_start,
                concatOutput->m_stride,
                concatOutput->m_part_size,
                concatOutput->m_end,
                concatOutput->m_axis,
            };
        }
        if (const auto& bodyOutput = std::dynamic_pointer_cast<NodeOp::BodyOutputDescription>(loop_output_desc)) {
            size_t iterations;
            if (bodyOutput->m_iteration == -1) {
                iterations = num_iterations_ - 1;
            } else {
                iterations = bodyOutput->m_iteration;
            }
            if (iterations_results_map_.count(iterations) == 0) {
                iterations_results_map_[iterations] = std::vector<uint64_t>{};
            }
            iterations_results_map_[iterations].push_back(bodyOutput->m_body_value_index);
        }
    }
    max_threads_per_block_ = context.device().props().maxThreadsPerBlock;

    for (const auto& [inputIdx, portMap] : portmap_inputs_) {
        const auto inputShape = inputs_info_[inputIdx].shape_;
        const auto inputType = inputs_info_[inputIdx].type_;

        kernel::Type_t element_type = convertDataType<ov::nvidia_gpu::kernel::Type_t>(inputType);
        kernel::Slice::Props props;
        std::copy(inputShape.begin(), inputShape.end(), props.old_shape);
        std::copy(inputShape.begin(), inputShape.end(), props.new_shape);
        props.axe = portMap.axis;
        props.new_shape[props.axe] = portMap.part_size;
        kernelmap_inputs_.emplace(inputIdx, kernel::Slice(element_type, props, max_threads_per_block_));
    }

    for (const auto& [resultIdx, outputIdx] : results_outputs_map_) {
        if (portmap_outputs_.count(outputIdx) > 0) {
            const auto& resultShape = results_info_[resultIdx].shape_;
            const auto outputShape = outputs_info_[outputIdx].shape_;
            const auto outputType = outputs_info_[outputIdx].type_;
            const auto portMap = portmap_outputs_.at(outputIdx);

            kernel::Type_t element_type = convertDataType<kernel::Type_t>(outputType);
            kernel::Insert::Props props;
            std::copy(resultShape.begin(), resultShape.end(), props.old_shape);
            std::copy(outputShape.begin(), outputShape.end(), props.new_shape);
            props.axe = portMap.axis;
            kernelmap_outputs_.emplace(outputIdx, kernel::Insert(element_type, props, max_threads_per_block_));
        }
    }

    updateExecSequence();

    slices_.reserve(portmap_inputs_.size());
    // Input mapping of ports
    for (auto& it : portmap_inputs_) {
        const auto& inputIdx = it.first;
        const auto& paramIdx = inputs_parameters_map_.at(inputIdx);
        slices_.emplace_back(*this, inputIdx, paramIdx);
    }

    transfers_.reserve(results_parameters_map_.size());
    // Back-edge mapping
    for (auto& [resultIdx, paramIdx] : results_parameters_map_) {
        transfers_.emplace_back(*this, resultIdx, paramIdx);
    }

    inserts_.reserve(results_outputs_map_.size());
    // Output mapping of ports
    for (const auto& [resultIdx, outputIdx] : results_outputs_map_) {
        if (portmap_outputs_.count(outputIdx) > 0) {
            inserts_.emplace_back(*this, resultIdx, outputIdx);
        }
    }
}

void TensorIteratorOp::Execute(const InferenceRequestContext& context,
                               Inputs inputTensors,
                               Outputs outputTensors,
                               const Workbuffers& workbuffers) const {
    const auto& stream = context.getThreadContext().stream();
    const auto& memoryManager = *memory_manager_;
    auto& mutableBuffer = workbuffers.mutable_buffers.at(0);
    // auto& cancellationToken = context.getCancellationToken();
    auto& executionDelegator = context.getExecutionDelegator();
    executionDelegator.set_stream(stream);

    // First iteration
    for (const auto inputIdx : invariant_inputs_) {
        const auto paramIdx = inputs_parameters_map_.at(inputIdx);
        transferParam(stream, mutableBuffer, inputTensors, 0, inputIdx, paramIdx);
    }
    for (const auto& [inputIdx, paramIdx] : inputs_parameters_map_) {
        if (portmap_inputs_.count(inputIdx) == 0) {
            transferParam(stream, mutableBuffer, inputTensors, 0, inputIdx, paramIdx);
        }
    }

    for (int64_t iter = 0; iter < num_iterations_; ++iter) {
        // Input mapping of ports
        for (const auto& slice : slices_) {
            slice(stream, inputTensors, mutableBuffer, iter);
        }

        // Inner loop
        executionDelegator.execute_sequence(this, memoryManager, mutableBuffer, context);

        // Back-edge mapping
        for (const auto& transfer : transfers_) {
            transfer(stream, mutableBuffer);
        }

        // Output mapping of ports
        for (const auto& insert : inserts_) {
            insert(stream, mutableBuffer, outputTensors, iter);
        }
    }

    // Copy data to output
    if (iterations_results_map_.count(num_iterations_ - 1) > 0) {
        for (const auto& resultIdx : iterations_results_map_.at(num_iterations_ - 1)) {
            const auto& outputIdx = results_outputs_map_.at(resultIdx);
            transferResult(stream, mutableBuffer, outputTensors, num_iterations_ - 1, resultIdx, outputIdx);
        }
    }
}

void TensorIteratorOp::ExecuteGraph(const InferenceRequestContext& context,
                                    Inputs inputTensors,
                                    Outputs outputTensors,
                                    const Workbuffers& workbuffers) {
    const auto& stream = context.getThreadContext().stream();
    const auto& memoryManager = *memory_manager_;
    const auto& mutableBuffer = workbuffers.mutable_buffers.at(0);

    // auto& graphContext = context.getCudaGraphContext();
    // const auto& opName = GetName();
    auto& tiGraphInfo = context.getCudaGraphContext().get_ti_graph(GetName());

    tiGraphInfo.launch_params_graph(stream);

    // OPENVINO_ASSERT(graphContext.get_kernels_count(opName) == slices_.size() + inserts_.size(),
    OPENVINO_ASSERT(tiGraphInfo.get_kernels_count() == slices_.size() + inserts_.size(),
                    "CudaGraphContext/TensorIteratorOp slices or inserts count incosistency");

    for (int64_t iter = 0; iter < num_iterations_; ++iter) {
        for (std::size_t i = 0; i < slices_.size(); ++i) {
            // graphContext.update_kernel(opName, i, slices_[i].get_knp(stream, mutableBuffer, inputTensors, iter));
            tiGraphInfo.update_kernel(i, slices_[i].get_knp(stream, mutableBuffer, inputTensors, iter));
        }
        for (std::size_t i = 0; i < inserts_.size(); ++i) {
            // graphContext.update_kernel(opName, i + slices_.size(), inserts_[i].get_knp(stream, mutableBuffer, outputTensors, iter));
            tiGraphInfo.update_kernel(i + slices_.size(), inserts_[i].get_knp(stream, mutableBuffer, outputTensors, iter));
        }
        // graphContext.launch_ti_graph(opName, stream);
        tiGraphInfo.launch_body_graph(stream);
    }

    tiGraphInfo.launch_results_graph(stream);
}

bool TensorIteratorOp::IsCudaGraphCompatible() const {
    return SubGraph::IsCudaGraphCompatible();
}

void TensorIteratorOp::Capture(InferenceRequestContext& context,
                               Inputs inputTensors,
                               Outputs outputTensors,
                               const Workbuffers& workbuffers) const {
    const auto& stream = context.getThreadContext().stream();
    const auto& memoryManager = *memory_manager_;
    auto& mutableBuffer = workbuffers.mutable_buffers.at(0);
    auto& executionDelegator = context.getExecutionDelegator();
    executionDelegator.set_stream(stream);

    // auto& graphContext = context.getCudaGraphContext();
    // const auto& opName = GetName();
    // graphContext.start_ti_graph_addition(opName);
    auto& tiGraphInfo = context.getCudaGraphContext().get_ti_graph(GetName());

    // TODO: refactor

    CUDA::GraphCapture capture{stream};
    {
        auto scope = capture.getScope();
        // First iteration
        for (const auto inputIdx : invariant_inputs_) {
            const auto paramIdx = inputs_parameters_map_.at(inputIdx);
            transferParam(stream, mutableBuffer, inputTensors, 0, inputIdx, paramIdx);
        }
        for (const auto& [inputIdx, paramIdx] : inputs_parameters_map_) {
            if (portmap_inputs_.count(inputIdx) == 0) {
                transferParam(stream, mutableBuffer, inputTensors, 0, inputIdx, paramIdx);
            }
        }
    }
    tiGraphInfo.set_params_graph(capture.getGraph());

    // CUDA::GraphCapture bodyCapture{stream};
    {
        auto scope = capture.getScope();
        // Input mapping of ports
        for (auto& slice : slices_) {
            // graphContext.add_kernel(opName, stream, slice.get_knp(stream, mutableBuffer, inputTensors, 0));
            tiGraphInfo.add_kernel(stream, slice.get_knp(stream, mutableBuffer, inputTensors, 0));
        }

        // Inner loop
        executionDelegator.capture_sequence(this, memoryManager, mutableBuffer, context);

        // Back-edge mapping
        for (auto& transfer : transfers_) {
            // graphContext.add_transfer(opName,
            tiGraphInfo.add_transfer(stream,
                                     CUDA::DevicePointer<void*>{transfer.get_dst(mutableBuffer)},
                                     CUDA::DevicePointer<const void*>{transfer.get_src(mutableBuffer)},
                                     transfer.get_param_size());
        }

        // Output mapping of ports
        for (auto& insert : inserts_) {
            // graphContext.add_kernel(opName, stream, insert.get_knp(stream, mutableBuffer, outputTensors, 0));
            tiGraphInfo.add_kernel(stream, insert.get_knp(stream, mutableBuffer, outputTensors, 0));
        }
    }
    // const auto& graph = bodyCapture.getGraph();
    // graphContext.add_ti_graph(opName, graph);
    // tiGraphInfo.set_body_graph(graph);
    tiGraphInfo.set_body_graph(capture.getGraph());

    {
        auto scope = capture.getScope();
        // TODO: Hadle n-th iteration situation
        // Copy data to output
        if (iterations_results_map_.count(num_iterations_ - 1) > 0) {
            for (const auto& resultIdx : iterations_results_map_.at(num_iterations_ - 1)) {
                const auto& outputIdx = results_outputs_map_.at(resultIdx);
                transferResult(stream, mutableBuffer, outputTensors, num_iterations_ - 1, resultIdx, outputIdx);
            }
        }
    }
    tiGraphInfo.set_results_graph(capture.getGraph());
}

TensorIteratorOp::SliceLauncher::SliceLauncher(const TensorIteratorOp& ti,
                                            //    const CUDA::Stream& stream,
                                            //    const CUDA::DevicePointer<void*> mutableBuffer,
                                            //    const IOperationExec::Inputs& inputTensors,
                                               uint64_t inputIdx,
                                               uint64_t paramIdx)
    : input_idx_{inputIdx},
      param_{*ti.params_[paramIdx]},
      memory_manager_{*ti.memory_manager_},
      slice_{ti.kernelmap_inputs_.at(inputIdx)} {
    OPENVINO_ASSERT(ti.portmap_inputs_.count(inputIdx) != 0, "Node name: ", ti.GetName());
    const auto& portMap = ti.portmap_inputs_.at(input_idx_);
    const auto& inputShape = ti.inputs_info_[input_idx_].shape_;
    start_ = portMap.start < 0 ? inputShape[portMap.axis] + portMap.start  : portMap.start;
    stride_ = portMap.stride;
}

TensorIteratorOp::TransferLauncher::TransferLauncher(const TensorIteratorOp& ti,
                                                    //  const CUDA::Stream& stream,
                                                    //  CUDA::DevicePointer<void*> mutableBuffer,
                                                     uint64_t resultIdx,
                                                     uint64_t paramIdx)
        : param_{*ti.params_[paramIdx]},
          result_{*ti.results_[resultIdx]},
          memory_manager_{*ti.memory_manager_} {
    param_size_ = ti.params_info_[paramIdx].size_;
    const auto resultSize = ti.results_info_[resultIdx].size_;
    OPENVINO_ASSERT(param_size_ == resultSize, "Node name: ", ti.GetName());
}

TensorIteratorOp::InsertLauncher::InsertLauncher(const TensorIteratorOp& ti,
                                                //  const CUDA::Stream& stream,
                                                //  CUDA::DevicePointer<void*> mutableBuffer,
                                                //  const IOperationExec::Outputs& outputTensors,
                                                 const std::size_t resultIdx,
                                                 const std::size_t outputIdx)
    : output_idx_{outputIdx},
      result_{*ti.results_[resultIdx]},
      memory_manager_{*ti.memory_manager_},
      insert_{ti.kernelmap_outputs_.at(outputIdx)} {
    OPENVINO_ASSERT(ti.portmap_outputs_.count(outputIdx) != 0, "Node name: ", ti.GetName());
    const auto& portMap = ti.portmap_outputs_.at(output_idx_);
    const auto& outputShape = ti.outputs_info_[output_idx_].shape_;
    start_ = portMap.start < 0 ? outputShape[portMap.axis] + portMap.start : portMap.start;
    stride_ = portMap.stride;
}

WorkbufferRequest TensorIteratorOp::GetWorkBufferRequest() const {
    std::vector<WorkbufferRequest::size_in_bytes_t> immutable_sizes;
    immutable_sizes.reserve(kernelmap_inputs_.size() + kernelmap_outputs_.size());
    for (const auto& kernel_map : kernelmap_inputs_) {
        immutable_sizes.push_back(kernel_map.second.getImmutableWorkbufferSize());
    }
    for (const auto& kernel_map : kernelmap_outputs_) {
        immutable_sizes.push_back(kernel_map.second.getImmutableWorkbufferSize());
    }
    return {immutable_sizes, SubGraph::GetWorkBufferRequest().mutable_sizes};
}

void TensorIteratorOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    OPENVINO_ASSERT(buffers.size() == kernelmap_inputs_.size() + kernelmap_outputs_.size(), "Node name: ", GetName());
    unsigned nextBufferIdx = 0;
    for (auto& kernel_map : kernelmap_inputs_) {
        auto& slice = kernel_map.second;
        slice.setImmutableWorkbuffer(buffers[nextBufferIdx++].get());
    }
    for (auto& kernel_map : kernelmap_outputs_) {
        auto& insert = kernel_map.second;
        insert.setImmutableWorkbuffer(buffers[nextBufferIdx++].get());
    }
}

void TensorIteratorOp::transferParam(const CUDA::Stream& stream,
                                     const CUDA::DevicePointer<void*> mutableBuffer,
                                     const IOperationExec::Inputs& inputTensors,
                                     const std::int64_t iter,
                                     const uint64_t inputIdx,
                                     const uint64_t paramIdx) const {
    OPENVINO_ASSERT(portmap_inputs_.count(inputIdx) == 0, "Node name: ", GetName());
    auto& memoryManager = *memory_manager_;
    const std::size_t inputSize = inputs_info_[inputIdx].size_;
    const std::size_t paramSize = params_info_[paramIdx].size_;

    auto& input = inputTensors[inputIdx];
    const auto& param = params_[paramIdx];
    auto outputTensors = memoryManager.outputTensorPointers(*param, mutableBuffer);
    OPENVINO_ASSERT(inputSize == paramSize, "Node name: ", GetName());

    stream.transfer(outputTensors[0], input, inputSize);
}

void TensorIteratorOp::transferResult(const CUDA::Stream& stream,
                                      CUDA::DevicePointer<void*> mutableBuffer,
                                      const IOperationExec::Outputs& outputTensors,
                                      const std::int64_t iter,
                                      const std::size_t resultIdx,
                                      const std::size_t outputIdx) const {
    OPENVINO_ASSERT(portmap_outputs_.count(outputIdx) == 0, "Node name: ", GetName());
    auto& memoryManager = *memory_manager_;
    const auto resultSize = results_info_[resultIdx].size_;
    const std::size_t outputSize = outputs_info_[outputIdx].size_;

    const auto result = results_[resultIdx];
    auto inTensors = memoryManager.inputTensorPointers(*result, mutableBuffer);
    const auto output = outputTensors[outputIdx];
    OPENVINO_ASSERT(resultSize == outputSize, "Node name: ", GetName());

    stream.transfer(output, inTensors[0], outputSize);
}

void TensorIteratorOp::updateExecSequence() {
    std::vector<OperationBase::Ptr> newExecSequence;
    for (const auto& op : exec_sequence_) {
        if (!dynamic_cast<const ParameterOp*>(op.get()) && !dynamic_cast<const ResultOp*>(op.get())) {
            newExecSequence.emplace_back(op);
        }
    }
    exec_sequence_ = std::move(newExecSequence);
}

OPERATION_REGISTER(TensorIteratorOp, TensorIterator);

}  // namespace nvidia_gpu
}  // namespace ov
