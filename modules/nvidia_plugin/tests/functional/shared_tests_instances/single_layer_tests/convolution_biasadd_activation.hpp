// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <ie_common.h>

#include <cstddef>
#include <functional_test_utils/precision_utils.hpp>
#include <ie_precision.hpp>
#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/node.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>
#include <ov_models/builders.hpp>
#include <ov_models/utils/ov_helpers.hpp>
#include "openvino/opsets/opset1.hpp"
#include <openvino/op/util/attr_types.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <shared_test_classes/single_layer/activation.hpp>
#include <shared_test_classes/single_layer/convolution.hpp>
#include <shared_test_classes/single_layer/group_convolution.hpp>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace LayerTestsDefinitions {
using ov::test::utils::ActivationTypes;

// TODO: Consider to add bias shape in here too, instead of deriving it in test class.
//       That would allow test generator to use bias shape from model
typedef std::tuple<convLayerTestParamsSet,
                   ActivationTypes  // Activation
                   >
    convBAATestParamSet;

typedef std::tuple<groupConvLayerTestParamsSet,
                   ActivationTypes  // Activation
                   >
    groupConvBAATestParamSet;

template <typename TConvLayerTest>
struct ConvBAATraits {};

template <>
struct ConvBAATraits<ConvolutionLayerTest> {
    using ConvNode = ov::opset1::Convolution;
    using ConvParamSet = convLayerTestParamsSet;
    using ConvSpecParamsSet = convSpecificParams;
    using ConvBAAParamSet = convBAATestParamSet;
    static constexpr const char* name = "ConvolutionBiasAddActivationLayerTest";
};

template <>
struct ConvBAATraits<GroupConvolutionLayerTest> {
    using ConvNode = ov::opset1::GroupConvolution;
    using ConvParamSet = groupConvLayerTestParamsSet;
    using ConvSpecParamsSet = groupConvSpecificParams;
    using ConvBAAParamSet = groupConvBAATestParamSet;
    static constexpr const char* name = "GroupConvolutionBiasAddActivationLayerTest";
};

/**
 * @brief A template class to test either Convolution/GroupConvolution + BiasAdd + Activation or
 * Convolution/GroupConvolution + BiasAdd + Add + Activation sequence
 * This class isn't intended for testing of fusing itself, so after the test network creation,
 * it may contain either FusedConvolution/FusedGroupConvolution nodes or simply a sequence of initial
 * Convolution/GroupConvolution, Add and Activation nodes
 */
template <typename TConvLayerTest, bool HasAddNode = false>
class BasicConvolutionBiasAddActivationLayerTest
    : public testing::WithParamInterface<typename ConvBAATraits<TConvLayerTest>::ConvBAAParamSet>,
      virtual public LayerTestsUtils::LayerTestsCommon {
    static_assert(std::is_same_v<TConvLayerTest, ConvolutionLayerTest> ||
                      std::is_same_v<TConvLayerTest, GroupConvolutionLayerTest>,
                  "TConvLayerTest should be either ConvolutionLayerTest or GroupConvolutionLayerTest");

public:
    using Traits = ConvBAATraits<TConvLayerTest>;

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 2, -1, 1000, 1);
    }

    static std::string getTestCaseName(testing::TestParamInfo<typename Traits::ConvBAAParamSet> obj) {
        typename Traits::ConvParamSet convParamSet;
        ActivationTypes activation;
        std::tie(convParamSet, activation) = obj.param;

        std::ostringstream result;
        result << TConvLayerTest::getTestCaseName({convParamSet, obj.index}) << "_";
        result << "Activation="
               << (activation == ActivationTypes::None
                       ? "None"
                       : LayerTestsDefinitions::activationNames[activation]);
        return result.str();
    }

protected:
    void SetUp() override {
        typename Traits::ConvParamSet convParamSet;
        ActivationTypes activation;
        std::tie(convParamSet, activation) = this->GetParam();

        ov::element::Type ngNetPrc = ov::element::Type_t::undefined;
        ov::ParameterVector params;
        std::shared_ptr<typename Traits::ConvNode> convLayer;
        std::tie(ngNetPrc, params, convLayer) = setUpConvolutionTestParams(convParamSet);
        if (ngNetPrc == ov::element::f16) {
            this->threshold = 0.5;
        }

        auto biasShape = convLayer->get_output_shape(0);
        constexpr size_t channel_dim_index = 1;
        for (size_t i = 0; i < biasShape.size(); ++i) {
            if (i != channel_dim_index) biasShape[i] = 1;
        }
        auto biasLayer = std::make_shared<ov::op::v0::Constant>(ngNetPrc, biasShape, std::vector<float>{1});

        auto biasAddLayer = std::make_shared<ov::op::v1::Add>(convLayer, biasLayer);

        std::shared_ptr<ov::Node> lastNode;
        if constexpr (HasAddNode) {
            auto addParam = std::make_shared<ov::opset1::Parameter>(ngNetPrc, convLayer->get_output_shape(0));
            params.push_back(addParam);
            auto addLayer = std::make_shared<ov::op::v1::Add>(biasAddLayer, addParam);
            lastNode = addLayer;
        } else {
            lastNode = biasAddLayer;
        }
        if (activation != ActivationTypes::None) {
            lastNode = ngraph::builder::makeActivation(lastNode, ngNetPrc, activation);
        }

        ov::ResultVector results{std::make_shared<ov::opset1::Result>(lastNode)};
        function = std::make_shared<ngraph::Function>(results, params, Traits::name);
    }

    std::tuple<ov::element::Type, ov::ParameterVector, std::shared_ptr<typename Traits::ConvNode>>
    setUpConvolutionTestParams(const typename Traits::ConvParamSet& convParamsSet) {
        typename Traits::ConvSpecParamsSet convParams;
        constexpr size_t convParamsSize = std::tuple_size<typename Traits::ConvSpecParamsSet>();
        std::vector<size_t> inputShape;
        auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) =
            convParamsSet;
        InferenceEngine::SizeVector kernel = std::get<0>(convParams);
        InferenceEngine::SizeVector stride = std::get<1>(convParams);
        std::vector<ptrdiff_t> padBegin = std::get<2>(convParams);
        std::vector<ptrdiff_t> padEnd = std::get<3>(convParams);
        InferenceEngine::SizeVector dilation = std::get<4>(convParams);
        size_t convOutChannels = std::get<5>(convParams);
        ov::op::PadType padType = std::get<convParamsSize - 1>(convParams);
        size_t numGroups = 0;
        constexpr bool isGroup = std::is_same_v<TConvLayerTest, GroupConvolutionLayerTest>;
        if constexpr (isGroup) {
            numGroups = std::get<6>(convParams);
        }

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        std::vector<float> filter_weights;

        std::shared_ptr<ov::Node> convNode = nullptr;
        if constexpr (!isGroup) {
            convNode = ngraph::builder::makeConvolution(params[0],
                                                        ngPrc,
                                                        kernel,
                                                        stride,
                                                        padBegin,
                                                        padEnd,
                                                        dilation,
                                                        padType,
                                                        convOutChannels,
                                                        false,
                                                        filter_weights);
        } else {
            convNode = ngraph::builder::makeGroupConvolution(params[0],
                                                             ngPrc,
                                                             kernel,
                                                             stride,
                                                             padBegin,
                                                             padEnd,
                                                             dilation,
                                                             padType,
                                                             convOutChannels,
                                                             numGroups,
                                                             false,
                                                             filter_weights);
        }
        return std::make_tuple(ngPrc, params, std::dynamic_pointer_cast<typename Traits::ConvNode>(convNode));
    }
};

using ConvolutionBiasAddActivationLayerTest = BasicConvolutionBiasAddActivationLayerTest<ConvolutionLayerTest, false>;
using GroupConvolutionBiasAddActivationLayerTest =
    BasicConvolutionBiasAddActivationLayerTest<GroupConvolutionLayerTest, false>;

using ConvolutionBiasAddAddActivationLayerTest = BasicConvolutionBiasAddActivationLayerTest<ConvolutionLayerTest, true>;
using GroupConvolutionBiasAddAddActivationLayerTest =
    BasicConvolutionBiasAddActivationLayerTest<GroupConvolutionLayerTest, true>;

}  // namespace LayerTestsDefinitions
