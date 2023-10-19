// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "details/cuda_type_traits.hpp"
#include "details/error.hpp"
#include "details/tensor_helpers.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

class Slice {
public:
    struct Props {
        Shape<size_t, 5> old_shape{};
        Shape<size_t, 5> new_shape{};
        size_t axe;
    };

    struct Params {
        inline const cudaKernelNodeParams* getKnp() {
            knp_.func = kernel;
            knp_.gridDim = num_blocks;
            knp_.blockDim = threads_per_block;
            knp_.sharedMemBytes = 0;
            args_[0] = &props;
            args_[1] = &start;
            args_[2] = &size;
            args_[3] = &x;
            args_[4] = &y;
            knp_.kernelParams = &args_[0];
            knp_.extra = nullptr;
            return &knp_;
        }

        // inline operator==(const Params& rhs) {
        //     return kernel == rhs.kernel &&
        //            num_blocks == rhs.num_blocks &&
        //            threads_per_block == rhs.threads_per_block &&
        //            props == rhs.props &&
        //            start == rhs.start &&
        //            size == rhs.size &&
        //            x == rhs.x &&
        //            y == rhs.y &&
        //            knp == rhs.knp_
        // }

        void* kernel;
        size_t num_blocks;
        size_t threads_per_block;
        const Props* props;
        size_t start;
        size_t size;
        const void* x;
        void* y;

    private:
        void* args_[5];
        cudaKernelNodeParams knp_;
    };

    Slice(Type_t element_type, const Props& props, size_t max_threads_per_block);
    Slice(Slice&&) = default;
    Slice& operator=(Slice&&) = default;

    void operator()(cudaStream_t stream, const void* src, void* dst, size_t start) const;

    std::unique_ptr<Params> getParams(const void* src, void* dst, const size_t start) const;

    size_t getImmutableWorkbufferSize() const;
    void setImmutableWorkbuffer(void* immutableBuffer);

private:
    template <typename T>
    void call(cudaStream_t stream, const void* src, void* dst, size_t start) const;

    Type_t element_type_{};
    Props props_{};
    size_t size_{};
    unsigned num_blocks_{};
    unsigned threads_per_block_{};
    void* props_ptr_{};
};

inline size_t Slice::getImmutableWorkbufferSize() const { return sizeof(props_); }

inline void Slice::setImmutableWorkbuffer(void* immutableBuffer) {
    kernel::throwIfError(
        cudaMemcpyAsync(immutableBuffer, static_cast<const void*>(&props_), sizeof(props_), cudaMemcpyHostToDevice));
    props_ptr_ = immutableBuffer;
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
