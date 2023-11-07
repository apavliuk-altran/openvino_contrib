// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/graph.hpp>

#include "cuda_tensor_mapping_context.hpp"

namespace ov {
namespace nvidia_gpu {

class TiCudaGraphInfo {
public:
    void add_transfer(const CUDA::Stream& stream,
                        CUDA::DevicePointer<void*> dst,
                        CUDA::DevicePointer<const void*> src,
                        std::size_t size);

    void add_kernel(const CUDA::Stream& stream, const cudaKernelNodeParams& knp);

    void set_params_graph(const CUDA::Graph& graph);
    void set_body_graph(const CUDA::Graph& graph);
    void set_results_graph(const CUDA::Graph& graph);

    // bool is_initialized() const;

    // void update_capture(const TensorMappingContext& context);
    void update_kernel(std::size_t index, const cudaKernelNodeParams& knp);

    void launch_params_graph(const CUDA::Stream& stream) const;
    void launch_body_graph(const CUDA::Stream& stream) const;
    void launch_results_graph(const CUDA::Stream& stream) const;

    std::size_t get_transfers_count() const;
    std::size_t get_kernels_count() const;

    // friend bool operator==(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs);
    // friend bool operator!=(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs);

private:
    std::optional<CUDA::Graph> paramsGraph_{};
    std::optional<CUDA::GraphExec> paramsGraphExec_{};

    std::optional<CUDA::Graph> bodyGraph_{};
    std::optional<CUDA::GraphExec> bodyGraphExec_{};

    std::optional<CUDA::Graph> resultsGraph_{};
    std::optional<CUDA::GraphExec> resultsGraphExec_{};

    std::vector<CUDA::TransferNode> transferNodes_;
    std::vector<CUDA::KernelNode> kernelNodes_;
};

class CudaGraphContext {
public:
    void reset();

    void start_next_graph_addition();

    // void start_ti_graph_addition(const std::string& ti_op_name);

    void add_parameter(const std::string& tensorName,
                       const CUDA::Stream& stream,
                       CUDA::DevicePointer<void*> dst,
                       const void* src,
                       std::size_t size);

    void add_result(const std::string& tensorName,
                    const CUDA::Stream& stream,
                    void* dst,
                    CUDA::DevicePointer<const void*> src,
                    std::size_t size);

    // void add_transfer(const std::string& ti_op_name,
    //                   const CUDA::Stream& stream,
    //                   CUDA::DevicePointer<void*> dst,
    //                   CUDA::DevicePointer<const void*> src,
    //                   std::size_t size);

    // void add_kernel(const std::string& ti_op_name,
    //                 const CUDA::Stream& stream,
    //                 const cudaKernelNodeParams& knp);

    void add_graph(const CUDA::Graph& graph);

    void add_ti_graph(const std::string& ti_op_name, const CUDA::Graph& graph);

    // const TiCudaGraphInfo& get_ti_graph(const std::string& ti_op_name) const;
    TiCudaGraphInfo& get_ti_graph(const std::string& ti_op_name) const;

    bool is_initialized() const;

    void update_capture(const TensorMappingContext& context);
    // void update_kernel(const std::string& ti_op_name,
    //                    std::size_t index,
    //                    const cudaKernelNodeParams& knp) const;

    void launch(std::size_t index, const CUDA::Stream& stream) const;
    // void launch_ti_graph(const std::string& ti_op_name, const CUDA::Stream& stream) const;

    std::size_t get_params_count() const;
    std::size_t get_results_count() const;

    // std::size_t get_transfers_count(const std::string& ti_op_name) const;
    // std::size_t get_kernels_count(const std::string& ti_op_name) const;

    std::size_t get_graphs_count() const;

    friend bool operator==(const CudaGraphContext& lhs, const CudaGraphContext& rhs);
    friend bool operator!=(const CudaGraphContext& lhs, const CudaGraphContext& rhs);

private:
    class CudaGraphInfo {
    public:
        void add_parameter(const std::string& tensorName,
                           const CUDA::Stream& stream,
                           CUDA::DevicePointer<void*> dst,
                           const void* src,
                           std::size_t size);

        void add_result(const std::string& tensorName,
                        const CUDA::Stream& stream,
                        void* dst,
                        CUDA::DevicePointer<const void*> src,
                        std::size_t size);

        // void add_transfer(const CUDA::Stream& stream,
        //                   CUDA::DevicePointer<void*> dst,
        //                   CUDA::DevicePointer<const void*> src,
        //                   std::size_t size);

        // void add_kernel(const CUDA::Stream& stream, const cudaKernelNodeParams& knp);

        void set_graph(const CUDA::Graph& graph);

        bool is_initialized() const;

        void update_capture(const TensorMappingContext& context);
        // void update_kernel(std::size_t index, const cudaKernelNodeParams& knp);

        void launch(const CUDA::Stream& stream) const;

        std::size_t get_params_count() const;
        std::size_t get_results_count() const;

        // std::size_t get_transfers_count() const;
        // std::size_t get_kernels_count() const;

        friend bool operator==(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs);
        friend bool operator!=(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs);

    private:
        std::optional<CUDA::Graph> graph_{};
        std::optional<CUDA::GraphExec> graphExec_{};
        std::map<std::string, CUDA::UploadNode> parameterNodes_;
        std::map<std::string, CUDA::DownloadNode> resultNodes_;

        // std::vector<CUDA::TransferNode> transferNodes_;
        // std::vector<CUDA::KernelNode> kernelNodes_;
    };

    friend bool operator==(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs);
    friend bool operator!=(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs);

private:
    std::vector<CudaGraphInfo> graphs_{};
    mutable std::unordered_map<std::string, TiCudaGraphInfo> ti_graphs_;
    // std::unordered_map<std::string, TiCudaGraphInfo> ti_graphs_;
    mutable std::size_t currentGraphIndex_ = 0;
};

bool operator==(const CudaGraphContext::CudaGraphInfo& lhs, const CudaGraphContext::CudaGraphInfo& rhs);

bool operator!=(const CudaGraphContext::CudaGraphInfo& lhs, const CudaGraphContext::CudaGraphInfo& rhs);

bool operator==(const CudaGraphContext& lhs, const CudaGraphContext& rhs);

bool operator!=(const CudaGraphContext& lhs, const CudaGraphContext& rhs);

}  // namespace nvidia_gpu
}  // namespace ov
