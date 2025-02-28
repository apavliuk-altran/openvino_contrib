// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/squeeze_unsqueeze.hpp"

#include <cuda_test_constants.hpp>
#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using ov::test::utils::SqueezeOpType;

namespace {

std::map<std::vector<size_t>, std::vector<std::vector<int>>> axesVectors = {
    {{1, 1, 1, 1},
     {{-1}, {0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {2, 3}, {0, 1, 2}, {0, 2, 3}, {1, 2, 3}, {0, 1, 2, 3}}},
    {{1, 2, 3, 4}, {{0}}},
    {{2, 1, 3, 4}, {{1}}},
    {{1}, {{-1}, {0}}},
    {{1, 2}, {{0}}},
    {{2, 1}, {{1}, {-1}}},
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<SqueezeOpType> opTypes = {SqueezeOpType::SQUEEZE,
                                                             SqueezeOpType::UNSQUEEZE};

INSTANTIATE_TEST_CASE_P(smoke_Basic,
                        SqueezeUnsqueezeLayerTest,
                        ::testing::Combine(::testing::ValuesIn(ov::test::utils::combineParams(axesVectors)),
                                           ::testing::ValuesIn(opTypes),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
                        SqueezeUnsqueezeLayerTest::getTestCaseName);
}  // namespace
