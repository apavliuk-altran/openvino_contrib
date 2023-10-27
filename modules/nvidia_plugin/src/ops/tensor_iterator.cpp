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

    // std::vector<SliceLauncher> slices;
    // slices.clear();
    slices_.reserve(portmap_inputs_.size());
    // Input mapping of ports
    for (auto& it : portmap_inputs_) {
        const auto& inputIdx = it.first;
        const auto& paramIdx = inputs_parameters_map_.at(inputIdx);
        // slices_.emplace_back(*this, stream, mutableBuffer, inputTensors, inputIdx, paramIdx);
        // slices.emplace_back(*this, mutableBuffer, inputTensors, inputIdx, paramIdx);
        slices_.emplace_back(*this, inputIdx, paramIdx);
    }

    // std::vector<TransferLauncher> transfers;
    // transfers.clear();
    transfers_.reserve(results_parameters_map_.size());
    // Back-edge mapping
    for (auto& [resultIdx, paramIdx] : results_parameters_map_) {
        // copyBackEdge(stream, mutableBuffer, resultIdx, paramIdx);
        // transfers_.emplace_back(*this, stream, mutableBuffer, resultIdx, paramIdx);
        // transfers.emplace_back(*this, mutableBuffer, resultIdx, paramIdx);
        transfers_.emplace_back(*this, resultIdx, paramIdx);
    }

    // std::vector<InsertLauncher> inserts;
    // inserts.clear();
    inserts_.reserve(results_outputs_map_.size());
    // Output mapping of ports
    for (const auto& [resultIdx, outputIdx] : results_outputs_map_) {
        if (portmap_outputs_.count(outputIdx) > 0) {
            // insertResult(stream, mutableBuffer, outputTensors, iter, resultIdx, outputIdx);
            // inserts_.emplace_back(*this, stream, mutableBuffer, outputTensors, resultIdx, outputIdx);
            // inserts.emplace_back(*this, mutableBuffer, outputTensors, resultIdx, outputIdx);
            inserts_.emplace_back(*this, resultIdx, outputIdx);
        }
    }

}

void TensorIteratorOp::Execute(const InferenceRequestContext& context,
                               Inputs inputTensors,
                               Outputs outputTensors,
                               const Workbuffers& workbuffers) const {
    const auto& stream = context.getThreadContext().stream();

    // std::cout << "---------------------------------------------------------------------------------------\n";
    // std::cout << "TensorIteratorOp::Execute()\n";
    // std::cout << "stream: " << stream.get() << ", this: " << this << '\n';
    // std::cout << "---------------------------------------------------------------------------------------\n";

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

    // std::vector<SliceLauncher> slices;
    // slices.clear();
    // slices.reserve(portmap_inputs_.size());
    // // Input mapping of ports
    // for (auto& it : portmap_inputs_) {
    //     const auto& inputIdx = it.first;
    //     const auto& paramIdx = inputs_parameters_map_.at(inputIdx);
    //     // slices_.emplace_back(*this, stream, mutableBuffer, inputTensors, inputIdx, paramIdx);
    //     // slices.emplace_back(*this, mutableBuffer, inputTensors, inputIdx, paramIdx);
    //     slices.emplace_back(*this, inputIdx, paramIdx);
    // }

    // std::vector<TransferLauncher> transfers;
    // transfers.clear();
    // transfers.reserve(results_parameters_map_.size());
    // // Back-edge mapping
    // for (auto& [resultIdx, paramIdx] : results_parameters_map_) {
    //     // copyBackEdge(stream, mutableBuffer, resultIdx, paramIdx);
    //     // transfers_.emplace_back(*this, stream, mutableBuffer, resultIdx, paramIdx);
    //     // transfers.emplace_back(*this, mutableBuffer, resultIdx, paramIdx);
    //     transfers.emplace_back(*this, resultIdx, paramIdx);
    // }

    // std::vector<InsertLauncher> inserts;
    // inserts.clear();
    // inserts.reserve(results_outputs_map_.size());
    // // Output mapping of ports
    // for (const auto& [resultIdx, outputIdx] : results_outputs_map_) {
    //     if (portmap_outputs_.count(outputIdx) > 0) {
    //         // insertResult(stream, mutableBuffer, outputTensors, iter, resultIdx, outputIdx);
    //         // inserts_.emplace_back(*this, stream, mutableBuffer, outputTensors, resultIdx, outputIdx);
    //         // inserts.emplace_back(*this, mutableBuffer, outputTensors, resultIdx, outputIdx);
    //         inserts.emplace_back(*this, resultIdx, outputIdx);
    //     }
    // }

    // for (int64_t iter = 0; iter < num_iterations_; ++iter) {

        // // Input mapping of ports
        // for (auto& it : portmap_inputs_) {
        //     const auto& inputIdx = it.first;
        //     const auto& paramIdx = inputs_parameters_map_.at(inputIdx);
        //     sliceParam(stream, mutableBuffer, inputTensors, iter, inputIdx, paramIdx);
        // }

        // // Inner loop
        // executionDelegator.execute_sequence(this, memoryManager, mutableBuffer, context);

        // // Back-edge mapping
        // for (auto& [resultIdx, paramIdx] : results_parameters_map_) {
        //     copyBackEdge(stream, mutableBuffer, resultIdx, paramIdx);
        // }

        // // Output mapping of ports
        // for (const auto& [resultIdx, outputIdx] : results_outputs_map_) {
        //     if (portmap_outputs_.count(outputIdx) > 0) {
        //         insertResult(stream, mutableBuffer, outputTensors, iter, resultIdx, outputIdx);
        //     }
        // }

        // // Copy data to output
        // if (iterations_results_map_.count(iter) > 0) {
        //     for (const auto& resultIdx : iterations_results_map_.at(iter)) {
        //         const auto& outputIdx = results_outputs_map_.at(resultIdx);
        //         transferResult(stream, mutableBuffer, outputTensors, iter, resultIdx, outputIdx);
        //     }
        // }
    // }

    for (int64_t iter = 0; iter < num_iterations_; ++iter) {

        // // Input mapping of ports
        for (const auto& slice : slices_) {
            // slice(iter);
            // slice(stream, iter);
            slice(stream, inputTensors, mutableBuffer, iter);
        }

        // Inner loop
        executionDelegator.execute_sequence(this, memoryManager, mutableBuffer, context);

        // // Back-edge mapping
        for (const auto& transfer : transfers_) {
            // transfer();
            // transfer(stream);
            transfer(stream, mutableBuffer);
        }

        // // Output mapping of ports
        for (const auto& insert : inserts_) {
            // insert(iter);
            // insert(stream, iter);
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

void TensorIteratorOp::ExecuteGraph(InferenceRequestContext& context,
                                    Inputs inputTensors,
                                    Outputs outputTensors,
                                    const Workbuffers& workbuffers) {
    // const auto& stream = context.getThreadContext().stream();
    // for (int64_t iter = 0; iter < num_iterations_; ++iter) {
    //     const_cast<InferenceRequestContext&>(context).getCudaGraphContext().update_capture(context.getTensorMappingContext());
    //     context.getCudaGraphContext().launch(graphIndex, stream);
    // }

    // std::cout << "---------------------------------------------------------------------------------------\n";
    // std::cout << "TensorIteratorOp::ExecuteGraph()\n";
    // std::cout << "---------------------------------------------------------------------------------------\n";

    const auto& stream = context.getThreadContext().stream();
    const auto& memoryManager = *memory_manager_;
    const auto& mutableBuffer = workbuffers.mutable_buffers.at(0);

    // auto& cancellationToken = context.getCancellationToken();
    // auto& executionDelegator = context.getExecutionDelegator();
    // executionDelegator.set_stream(stream);

    // TODO: refactor
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

    auto& graphContext = context.getCudaGraphContext();
    for (int64_t iter = 0; iter < num_iterations_; ++iter) {
        // for (auto& slice : slices_) {
        //     slice.update_capture(*graph_exec_, iter);
        // }
        // for (auto& insert : inserts_) {
        //     insert.update_capture(*graph_exec_, iter);
        // }
        // graph_exec_->launch(tream);

        for (std::size_t i = 0; i < slices_.size(); ++i) {
            graphContext.update_slice(i, slices_[i].get_params(stream, mutableBuffer, inputTensors, iter));
        }
        for (std::size_t i = 0; i < inserts_.size(); ++i) {
            graphContext.update_insert(i, inserts_[i].get_params(stream, mutableBuffer, outputTensors, iter));
        }
        graphContext.launch_ti_graph(stream);
    }

    // TODO: Hadle n-th iteration situation
    // Copy data to output
    if (iterations_results_map_.count(num_iterations_ - 1) > 0) {
        for (const auto& resultIdx : iterations_results_map_.at(num_iterations_ - 1)) {
            const auto& outputIdx = results_outputs_map_.at(resultIdx);
            transferResult(stream, mutableBuffer, outputTensors, num_iterations_ - 1, resultIdx, outputIdx);
        }
    }
}

// TODO: Investigate problem with multi-graphs in some networks
// benchmark_app may hang in throughput mode

// bool TensorIteratorOp::IsCudaGraphCompatible() const { return false; }

bool TensorIteratorOp::IsCudaGraphCompatible() const {
    return SubGraph::IsCudaGraphCompatible();
}

void TensorIteratorOp::Capture(InferenceRequestContext& context,
                               Inputs inputTensors,
                               Outputs outputTensors,
                               const Workbuffers& workbuffers) const {
    // std::cout << "---------------------------------------------------------------------------------------\n";
    // std::cout << "TensorIteratorOp::Capture()\n";
    // std::cout << "---------------------------------------------------------------------------------------\n";

    // Execute(context, inputTensors, outputTensors, workbuffers);
    const auto& stream = context.getThreadContext().stream();
    const auto& memoryManager = *memory_manager_;
    auto& mutableBuffer = workbuffers.mutable_buffers.at(0);
    // auto& cancellationToken = context.getCancellationToken();
    auto& executionDelegator = context.getExecutionDelegator();
    executionDelegator.set_stream(stream);

    // slices_.clear();
    // slices_.reserve(portmap_inputs_.size());
    // // Input mapping of ports
    // for (auto& it : portmap_inputs_) {
    //     const auto& inputIdx = it.first;
    //     const auto& paramIdx = inputs_parameters_map_.at(inputIdx);
    //     // slices_.emplace_back(*this, stream, mutableBuffer, inputTensors, inputIdx, paramIdx);
    //     // slices_.emplace_back(*this, mutableBuffer, inputTensors, inputIdx, paramIdx);
    //     slices_.emplace_back(*this, inputIdx, paramIdx);
    // }

    // transfers_.clear();
    // transfers_.reserve(results_parameters_map_.size());
    // // Back-edge mapping
    // for (auto& [resultIdx, paramIdx] : results_parameters_map_) {
    //     // copyBackEdge(stream, mutableBuffer, resultIdx, paramIdx);
    //     // transfers_.emplace_back(*this, stream, mutableBuffer, resultIdx, paramIdx);
    //     // transfers_.emplace_back(*this, mutableBuffer, resultIdx, paramIdx);
    //     transfers_.emplace_back(*this, resultIdx, paramIdx);
    // }

    // inserts_.clear();
    // inserts_.reserve(results_outputs_map_.size());
    // // Output mapping of ports
    // for (const auto& [resultIdx, outputIdx] : results_outputs_map_) {
    //     if (portmap_outputs_.count(outputIdx) > 0) {
    //         // insertResult(stream, mutableBuffer, outputTensors, iter, resultIdx, outputIdx);
    //         // inserts_.emplace_back(*this, stream, mutableBuffer, outputTensors, resultIdx, outputIdx);
    //         // inserts_.emplace_back(*this, mutableBuffer, outputTensors, resultIdx, outputIdx);
    //         inserts_.emplace_back(*this, resultIdx, outputIdx);
    //     }
    // }

    auto& graphContext = context.getCudaGraphContext();
    graphContext.start_ti_graph_addition();
    CUDA::GraphCapture capture{stream};
    // cudaGraph_t cudaGraph{};
    {
        auto scope = capture.getScope();

        // throwIfError(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
        for (auto& slice : slices_) {
            // slice.capture();
            // slice.capture(stream);
            graphContext.add_slice(stream, slice.get_params(stream, mutableBuffer, inputTensors, 0));
        }

        // Inner loop
        executionDelegator.capture_sequence(this, memoryManager, mutableBuffer, context);

        // // Back-edge mapping
        for (auto& transfer : transfers_) {
            // transfer.capture();
            // transfer.capture(stream);
            graphContext.add_transfer(stream,
                                      CUDA::DevicePointer<void*>{transfer.get_dst(mutableBuffer)},
                                      CUDA::DevicePointer<const void*>{transfer.get_src(mutableBuffer)},
                                      transfer.get_param_size());
        }

        // // Output mapping of ports
        for (auto& insert : inserts_) {
            // insert.capture();
            // insert.capture(stream);
            graphContext.add_insert(stream, insert.get_params(stream, mutableBuffer, outputTensors, 0));
        }
        // throwIfError(cudaStreamEndCapture(stream.get(), &cudaGraph));
    }
    const auto& graph = capture.getGraph();
    // graph_.emplace(graph);
    // graph_exec_.emplace(graph);
    // graph_.emplace(cudaGraph);
    // graph_exec_.emplace(cudaGraph);
    graphContext.add_ti_graph(graph);
}

TensorIteratorOp::SliceLauncher::SliceLauncher(const TensorIteratorOp& ti,
                                            //    const CUDA::Stream& stream,
                                            //    const CUDA::DevicePointer<void*> mutableBuffer,
                                            //    const IOperationExec::Inputs& inputTensors,
                                               uint64_t inputIdx,
                                               uint64_t paramIdx)
    // : stream_(stream),
    : input_idx_{inputIdx},
      param_{*ti.params_[paramIdx]},
      memory_manager_{*ti.memory_manager_},
      slice_{ti.kernelmap_inputs_.at(inputIdx)} {
    // std::cout << "---------------------------------------------------------------------------------------\n";
    // std::cout << "SliceLauncher::SliceLauncher()\n";

    OPENVINO_ASSERT(ti.portmap_inputs_.count(inputIdx) != 0, "Node name: ", ti.GetName());
    // const std::size_t inputSize = ti.inputs_info_[inputIdx].size_;
    // const std::size_t paramSize = ti.params_info_[paramIdx].size_;

    // const auto& param = ti.params_[paramIdx];

    // const auto& slice = kernelmap_inputs_.at(inputIdx);
    // std::size_t start;
    // start += iter * portMap.stride;
    const auto& portMap = ti.portmap_inputs_.at(input_idx_);
    const auto& inputShape = ti.inputs_info_[input_idx_].shape_;
    start_ = portMap.start < 0 ? inputShape[portMap.axis] + portMap.start  : portMap.start;
    stride_ = portMap.stride;

    // std::cout << "---------------------------------------------------------------------------------------\n";
}

TensorIteratorOp::TransferLauncher::TransferLauncher(const TensorIteratorOp& ti,
                                                    //  const CUDA::Stream& stream,
                                                    //  CUDA::DevicePointer<void*> mutableBuffer,
                                                     uint64_t resultIdx,
                                                     uint64_t paramIdx)
    // : stream_(stream) {
        : param_{*ti.params_[paramIdx]},
          result_{*ti.results_[resultIdx]},
          memory_manager_{*ti.memory_manager_}
    {
    // std::cout << "=======================================================================================\n";
    // std::cout << "TransferLauncher::TransferLauncher()\n";

    // auto& memoryManager = *ti.memory_manager_;
    // const auto& result = ti.results_[resultIdx];
    // const auto& param = ti.params_[paramIdx];
    // const std::size_t paramSize = ti.params_info_[paramIdx].size_;
    // OPENVINO_ASSERT(paramSize == resultSize, "Node name: ", ti.GetName());
    param_size_ = ti.params_info_[paramIdx].size_;
    const auto resultSize = ti.results_info_[resultIdx].size_;
    OPENVINO_ASSERT(param_size_ == resultSize, "Node name: ", ti.GetName());
    // count_ = paramSize;

    // stream.transfer(paramTensors[0], resultTensors[0], paramSize);

    // std::cout << "=======================================================================================\n";
}

TensorIteratorOp::InsertLauncher::InsertLauncher(const TensorIteratorOp& ti,
                                                //  const CUDA::Stream& stream,
                                                //  CUDA::DevicePointer<void*> mutableBuffer,
                                                //  const IOperationExec::Outputs& outputTensors,
                                                 const std::size_t resultIdx,
                                                 const std::size_t outputIdx)
    // : stream_(stream),
    : output_idx_{outputIdx},
      result_{*ti.results_[resultIdx]},
      memory_manager_{*ti.memory_manager_},
      insert_{ti.kernelmap_outputs_.at(outputIdx)} {
    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    // std::cout << "InsertLauncher::InsertLauncher()\n";

    OPENVINO_ASSERT(ti.portmap_outputs_.count(outputIdx) != 0, "Node name: ", ti.GetName());
    // auto& memoryManager = *ti.memory_manager_;
    // const auto resultSize = ti.results_info_[resultIdx].size_;
    // const std::size_t outputSize = ti.outputs_info_[outputIdx].size_;

    // const auto& result = ti.results_[resultIdx];
    const auto& portMap = ti.portmap_outputs_.at(output_idx_);
    const auto& outputShape = ti.outputs_info_[output_idx_].shape_;

    // const auto& insert = ti.kernelmap_outputs_.at(outputIdx);
    // std::size_t start;
    // if (portMap.start < 0) {
    //     start = outputShape[portMap.axis] + portMap.start;
    // } else {
    //     start = portMap.start;
    // }
    // start += iter * portMap.stride;
    // start_ = start;
    start_ = portMap.start < 0 ? outputShape[portMap.axis] + portMap.start : portMap.start;
    stride_ = portMap.stride;

    // insert(stream.get(), inputTensors[0].get(), output.get(), start);

    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
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
    // std::cout << "---------------------------------------------------------------------------------------\n";
    // std::cout << "TensorIteratorOp::transferParam()\n";

    OPENVINO_ASSERT(portmap_inputs_.count(inputIdx) == 0, "Node name: ", GetName());
    auto& memoryManager = *memory_manager_;
    const std::size_t inputSize = inputs_info_[inputIdx].size_;
    const std::size_t paramSize = params_info_[paramIdx].size_;

    auto& input = inputTensors[inputIdx];
    const auto& param = params_[paramIdx];
    auto outputTensors = memoryManager.outputTensorPointers(*param, mutableBuffer);
    OPENVINO_ASSERT(inputSize == paramSize, "Node name: ", GetName());

    stream.transfer(outputTensors[0], input, inputSize);

    // std::cout << "---------------------------------------------------------------------------------------\n";
}

void TensorIteratorOp::sliceParam(const CUDA::Stream& stream,
                                  const CUDA::DevicePointer<void*> mutableBuffer,
                                  const IOperationExec::Inputs& inputTensors,
                                  const std::int64_t iter,
                                  const uint64_t inputIdx,
                                  const uint64_t paramIdx) const {
    // std::cout << "---------------------------------------------------------------------------------------\n";
    // std::cout << "TensorIteratorOp::sliceParam()\n";

    OPENVINO_ASSERT(portmap_inputs_.count(inputIdx) != 0, "Node name: ", GetName());
    auto& memoryManager = *memory_manager_;
    const std::size_t inputSize = inputs_info_[inputIdx].size_;
    const std::size_t paramSize = params_info_[paramIdx].size_;

    const auto& portMap = portmap_inputs_.at(inputIdx);
    const auto& param = params_[paramIdx];
    auto outputTensors = memoryManager.outputTensorPointers(*param, mutableBuffer);
    const auto inputShape = inputs_info_[inputIdx].shape_;

    const auto& slice = kernelmap_inputs_.at(inputIdx);
    std::size_t start;
    if (portMap.start < 0) {
        start = inputShape[portMap.axis] + portMap.start;
    } else {
        start = portMap.start;
    }
    start += iter * portMap.stride;
    auto input = inputTensors[inputIdx];

    slice(stream.get(), input.get(), outputTensors[0].get(), start);

    // std::cout << "---------------------------------------------------------------------------------------\n";
}

void TensorIteratorOp::copyBackEdge(const CUDA::Stream& stream,
                                    CUDA::DevicePointer<void*> mutableBuffer,
                                    const uint64_t resultIdx,
                                    const uint64_t paramIdx) const {
    // std::cout << "=======================================================================================\n";
    // std::cout << "TensorIteratorOp::copyBackEdge()\n";

    auto& memoryManager = *memory_manager_;
    const auto& result = results_[resultIdx];
    const auto& param = params_[paramIdx];
    auto paramTensors = memoryManager.outputTensorPointers(*param, mutableBuffer);
    auto resultTensors = memoryManager.inputTensorPointers(*result, mutableBuffer);
    const std::size_t paramSize = params_info_[paramIdx].size_;
    const std::size_t resultSize = results_info_[resultIdx].size_;
    OPENVINO_ASSERT(paramSize == resultSize, "Node name: ", GetName());

    stream.transfer(paramTensors[0], resultTensors[0], paramSize);

    // std::cout << "=======================================================================================\n";
}

void TensorIteratorOp::transferResult(const CUDA::Stream& stream,
                                      CUDA::DevicePointer<void*> mutableBuffer,
                                      const IOperationExec::Outputs& outputTensors,
                                      const std::int64_t iter,
                                      const std::size_t resultIdx,
                                      const std::size_t outputIdx) const {
    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    // std::cout << "TensorIteratorOp::transferResult()\n";

    OPENVINO_ASSERT(portmap_outputs_.count(outputIdx) == 0, "Node name: ", GetName());
    auto& memoryManager = *memory_manager_;
    const auto resultSize = results_info_[resultIdx].size_;
    const std::size_t outputSize = outputs_info_[outputIdx].size_;

    const auto result = results_[resultIdx];
    auto inTensors = memoryManager.inputTensorPointers(*result, mutableBuffer);
    const auto output = outputTensors[outputIdx];
    OPENVINO_ASSERT(resultSize == outputSize, "Node name: ", GetName());

    stream.transfer(output, inTensors[0], outputSize);

    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
}

void TensorIteratorOp::insertResult(const CUDA::Stream& stream,
                                    CUDA::DevicePointer<void*> mutableBuffer,
                                    const IOperationExec::Outputs& outputTensors,
                                    const std::int64_t iter,
                                    const std::size_t resultIdx,
                                    const std::size_t outputIdx) const {
    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    // std::cout << "TensorIteratorOp::insertResult()\n";

    OPENVINO_ASSERT(portmap_outputs_.count(outputIdx) != 0, "Node name: ", GetName());
    auto& memoryManager = *memory_manager_;
    const auto resultSize = results_info_[resultIdx].size_;
    const std::size_t outputSize = outputs_info_[outputIdx].size_;

    auto output = outputTensors[outputIdx];
    const auto& result = results_[resultIdx];
    auto inputTensors = memoryManager.inputTensorPointers(*result, mutableBuffer);
    const auto portMap = portmap_outputs_.at(outputIdx);
    const auto outputShape = outputs_info_[outputIdx].shape_;

    const auto& insert = kernelmap_outputs_.at(outputIdx);
    std::size_t start;
    if (portMap.start < 0) {
        start = outputShape[portMap.axis] + portMap.start;
    } else {
        start = portMap.start;
    }
    start += iter * portMap.stride;

    insert(stream.get(), inputTensors[0].get(), output.get(), start);

    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
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
