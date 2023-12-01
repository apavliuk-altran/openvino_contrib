// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/graph.hpp>

#include "cuda_tensor_mapping_context.hpp"

namespace ov {
namespace nvidia_gpu {

// enum class GraphInfoType { INFO, CONTEXT };

// class ICudaGraphInfo : public std::enable_shared_from_this<ICudaGraphInfo> {
class ICudaGraphInfo {
public: 
    virtual ~ICudaGraphInfo() = 0;
    
    virtual void reset() = 0;

    virtual void add_parameter(const std::string& tensorName,
                       const CUDA::Stream& stream,
                       CUDA::DevicePointer<void*> dst,
                       const void* src,
                       std::size_t size) = 0;

    virtual void add_result(const std::string& tensorName,
                    const CUDA::Stream& stream,
                    void* dst,
                    CUDA::DevicePointer<const void*> src,
                    std::size_t size) = 0;

    virtual void add_transfer(const CUDA::Stream& stream,
                      CUDA::DevicePointer<void*> dst,
                      CUDA::DevicePointer<const void*> src,
                      std::size_t size) = 0;

    template <typename... Args>
    void add_kernel(const CUDA::Stream& stream, void* kernel, dim3 gridDim, dim3 blockDim, Args&&... args) {
        CUDA::CaptureInfo captureInfo{stream};
        get_kernels().emplace_back(captureInfo.addKernelNode(kernel, gridDim, blockDim, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void update_kernel(std::size_t index, Args&&... args) {
        get_kernels()[index].update_params(get_graph_exec().value(), std::forward<Args>(args)...);
    }

    virtual void set_current_graph(const CUDA::Graph& graph) = 0;

    virtual bool is_initialized() const = 0;
    virtual bool is_nested() const = 0;

    virtual void update_capture(const TensorMappingContext& context) = 0;

    // virtual void add(std::unique_ptr<ICudaGraphInfo> ptr) = 0;
    // virtual void add(std::shared_ptr<ICudaGraphInfo> ptr) = 0;
    virtual ICudaGraphInfo& add(std::shared_ptr<ICudaGraphInfo> ptr) = 0;

    // virtual const ICudaGraphInfo& get_current_graph() const = 0;
    virtual ICudaGraphInfo& get_current_graph() = 0;

    virtual void select_current_graph(std::size_t index) = 0;

    virtual std::size_t get_params_count() const = 0;
    virtual std::size_t get_results_count() const = 0;
    virtual std::size_t get_transfers_count() const = 0;
    virtual std::size_t get_kernels_count() const = 0;

    virtual std::size_t get_graphs_count() const = 0;

    // virtual void set_graph(const CUDA::Graph& graph) = 0;

    virtual void launch(const CUDA::Stream& stream) const = 0;

// protected:
    virtual std::vector<CUDA::KernelNode>& get_kernels() = 0;
    virtual std::optional<CUDA::GraphExec>& get_graph_exec() = 0;
};

inline ICudaGraphInfo::~ICudaGraphInfo() = default;

class CudaGraphInfo : public ICudaGraphInfo {
public:
    CudaGraphInfo() = default;
    CudaGraphInfo(const CudaGraphInfo&) = delete;
    CudaGraphInfo& operator=(const CudaGraphInfo&) = delete;

    // static std::unique_ptr<ICudaGraphInfo> create() {
    //     return std::make_unique<CudaGraphInfo>();
    // }
    static std::shared_ptr<ICudaGraphInfo> create() {
        // TODO: don't use make_shared?
        return std::make_shared<CudaGraphInfo>();
    }

    void reset() override;

    void add_parameter(const std::string& tensorName,
                       const CUDA::Stream& stream,
                       CUDA::DevicePointer<void*> dst,
                       const void* src,
                       std::size_t size) override;

    void add_result(const std::string& tensorName,
                    const CUDA::Stream& stream,
                    void* dst,
                    CUDA::DevicePointer<const void*> src,
                    std::size_t size) override;

    void add_transfer(const CUDA::Stream& stream,
                      CUDA::DevicePointer<void*> dst,
                      CUDA::DevicePointer<const void*> src,
                      std::size_t size) override;

    // template <typename... Args>
    // void add_kernel(const CUDA::Stream& stream, void* kernel, dim3 gridDim, dim3 blockDim, Args&&... args) {
    //     CUDA::CaptureInfo captureInfo{stream};
    //     kernelNodes_.emplace_back(captureInfo.addKernelNode(kernel, gridDim, blockDim, std::forward<Args>(args)...));
    // }

    // template <typename... Args>
    // void update_kernel(std::size_t index, Args&&... args) {
    //     kernelNodes_[index].update_params(graphExec_.value(), std::forward<Args>(args)...);
    // }

    void set_current_graph(const CUDA::Graph& graph) override {
        graph_.emplace(graph);
        graphExec_.emplace(graph);
    }

    bool is_initialized() const override;
    bool is_nested() const override { return false; };

    void update_capture(const TensorMappingContext& context) override;

    // void add(std::unique_ptr<ICudaGraphInfo> ptr) override {
    // void add(std::shared_ptr<ICudaGraphInfo> ptr) override {
    ICudaGraphInfo& add(std::shared_ptr<ICudaGraphInfo> ptr) override {
        OPENVINO_THROW("add() called for CudaGraphInfo");
    }

    // const ICudaGraphInfo& get_current_graph() const override { return *this; }
    ICudaGraphInfo& get_current_graph() override { return *this; }

    void select_current_graph(std::size_t index) override {
        OPENVINO_THROW("select_current_graph() called for CudaGraphInfo");
    }

    std::size_t get_params_count() const override { return parameterNodes_.size(); }
    std::size_t get_results_count() const override { return resultNodes_.size(); }
    std::size_t get_transfers_count() const override { return transferNodes_.size(); }
    std::size_t get_kernels_count() const override { return kernelNodes_.size(); }

    std::size_t get_graphs_count() const override;

    // void set_graph(const CUDA::Graph& graph) override;

    void launch(const CUDA::Stream& stream) const override;

    // friend bool operator==(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs);
    // friend bool operator!=(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs);

// protected:
    std::vector<CUDA::KernelNode>& get_kernels() override { return kernelNodes_; };
    std::optional<CUDA::GraphExec>& get_graph_exec() override { return graphExec_; };

private:
    std::optional<CUDA::Graph> graph_{};
    std::optional<CUDA::GraphExec> graphExec_{};

    std::map<std::string, CUDA::UploadNode> parameterNodes_;
    std::map<std::string, CUDA::DownloadNode> resultNodes_;

    std::vector<CUDA::TransferNode> transferNodes_;
    std::vector<CUDA::KernelNode> kernelNodes_;
};

class CudaGraphContext : public ICudaGraphInfo{
public:
    CudaGraphContext() = default;
    CudaGraphContext(const CudaGraphContext&) = delete;
    CudaGraphContext& operator=(const CudaGraphContext&) = delete;

    // static std::unique_ptr<ICudaGraphInfo> create() {
    //     return std::make_unique<CudaGraphContext>();
    // }
    static std::shared_ptr<ICudaGraphInfo> create() {
        return std::make_shared<CudaGraphContext>();
    }

    void reset() override;

    void add_parameter(const std::string& tensorName,
                       const CUDA::Stream& stream,
                       CUDA::DevicePointer<void*> dst,
                       const void* src,
                       std::size_t size) override;

    void add_result(const std::string& tensorName,
                    const CUDA::Stream& stream,
                    void* dst,
                    CUDA::DevicePointer<const void*> src,
                    std::size_t size) override;

    void add_transfer(const CUDA::Stream& stream,
                      CUDA::DevicePointer<void*> dst,
                      CUDA::DevicePointer<const void*> src,
                      std::size_t size) override;

    void set_current_graph(const CUDA::Graph& graph) override;

    bool is_initialized() const override;
    bool is_nested() const override { return true; };

    void update_capture(const TensorMappingContext& context) override;

    // void add(std::unique_ptr<ICudaGraphInfo> ptr) override;
    // void add(std::shared_ptr<ICudaGraphInfo> ptr) override;
    ICudaGraphInfo& add(std::shared_ptr<ICudaGraphInfo> ptr) override;

    // const ICudaGraphInfo& get_current_graph() const override;
    ICudaGraphInfo& get_current_graph() override;

    void select_current_graph(std::size_t index) override;

    std::size_t get_params_count() const override;
    std::size_t get_results_count() const override;
    std::size_t get_transfers_count() const override;
    std::size_t get_kernels_count() const override;

    std::size_t get_graphs_count() const override;

    // void set_graph(const CUDA::Graph& graph) override;

    void launch(const CUDA::Stream& stream) const override;

    // friend bool operator==(const CudaGraphContext& lhs, const CudaGraphContext& rhs);
    // friend bool operator!=(const CudaGraphContext& lhs, const CudaGraphContext& rhs);

// protected:
    std::vector<CUDA::KernelNode>& get_kernels() override { return graphs_[currentGraphIndex_]->get_kernels(); };
    std::optional<CUDA::GraphExec>& get_graph_exec() override { return graphs_[currentGraphIndex_]->get_graph_exec(); };

private:
    // std::vector<std::unique_ptr<ICudaGraphInfo>> graphs_{};
    std::vector<std::shared_ptr<ICudaGraphInfo>> graphs_{};
    std::size_t currentGraphIndex_ = 0;
};

// bool operator==(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs);

// bool operator!=(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs);

// bool operator==(const CudaGraphContext& lhs, const CudaGraphContext& rhs);

// bool operator!=(const CudaGraphContext& lhs, const CudaGraphContext& rhs);

}  // namespace nvidia_gpu
}  // namespace ov
