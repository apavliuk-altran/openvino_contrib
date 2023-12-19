// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

// #include "cuda_graph_topology_runner.hpp"
#include "cuda_eager_topology_runner.hpp"
#include "cuda_simple_execution_delegator.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/data_utils.hpp"
#include "ops/parameter.hpp"
#include "ops/result.hpp"

using namespace ov::nvidia_gpu;
using namespace testing;
using ov::test::utils::EltwiseTypes;

namespace {

constexpr int TO = 10;
// constexpr int FROM = -10;
constexpr int FROM = 0;
constexpr int SEED = 1;

constexpr std::size_t INPUTS_COUNT = 2;
constexpr int64_t CONCAT_AXIS = 0;

constexpr float THRESHOLD = 0.01f;

using CalcType = float;
constexpr auto CALC_ELEMENT_TYPE = ov::element::Type_t::f32;

inline CalcType* getMutablePtr(ov::Tensor& tensor) { return static_cast<CalcType*>(tensor.data()); }

inline const CalcType* getConstPtr(const ov::Tensor& tensor) {
    return static_cast<const CalcType*>(tensor.data());
}

void generateInput(ov::Tensor& tensor, int to = TO, int from = FROM, int seed = SEED) {
    EXPECT_EQ(tensor.get_element_type(), CALC_ELEMENT_TYPE);
    auto* ptr = getMutablePtr(tensor);
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dist(from, to);
    std::generate(ptr, ptr + tensor.get_size(), [&dist, &engine]() { return CalcType{dist(engine)}; });
}

void validateOutput(const ov::Tensor& tensor, const std::vector<CalcType>& refVector, float threshold) {
    EXPECT_EQ(tensor.get_element_type(), CALC_ELEMENT_TYPE);
    const auto size = tensor.get_size();
    EXPECT_EQ(size, refVector.size());
    const auto* ptr = getConstPtr(tensor);
    bool areEqual = std::equal(ptr, ptr + size, refVector.cbegin(), [threshold](auto val1, auto val2) {
        // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // std::cout << "val1  = " << static_cast<float>(val1) << ", val2 = " << static_cast<float>(val2) << '\n';
        // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        return std::abs(val1 - val2) < threshold;
    });
    EXPECT_TRUE(areEqual);
}

}  // namespace

// TODO: Get rid of ngraph::*
class GRUTI {
public:
    static std::shared_ptr<ov::Model> createNetwork() {
        // ov::element::Type prc = CALC_ELEMENT_TYPE;
        // ov::Shape shape{1, 2, 3, 4};
        // ov::ParameterVector params;
        // for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
        //     params.emplace_back(std::make_shared<ov::op::v0::Parameter>(prc, shape));
        // }
        // const auto add0 = ngraph::builder::makeEltwise(params[0], params[1], EltwiseTypes::ADD);
        // const auto add1 = ngraph::builder::makeEltwise(params[2], params[3], EltwiseTypes::ADD);

        // const auto mul = ngraph::builder::makeEltwise(add0, add1, EltwiseTypes::MULTIPLY);
        // const auto result = std::make_shared<ov::op::v0::Result>(mul);
        // return std::make_shared<ov::Model>(result, params, "AddMul");

        size_t seqLengths = 20;
        // bool should_decompose;
        size_t batch = 1;
        size_t hidden_size = 10;
        size_t inputSize = 10;
        size_t seqAxis = 1;
        // ngraph::helpers::TensorIteratorBody ti_body;
        float clip = 0.0;
        // ngraph::op::RecurrentSequenceDirection direction;
        // InferenceEngine::Precision netPrecision;
        ov::element::Type ngPrc = CALC_ELEMENT_TYPE;

        auto tensorIterator = std::make_shared<ngraph::opset5::TensorIterator>();
        auto axis = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1},
                                                               std::vector<int64_t>{static_cast<int64_t>(seqAxis)});
        std::vector<std::vector<size_t>> inputShapes = {
                {{batch, seqLengths, inputSize}, {batch, hidden_size}, {3 * hidden_size, inputSize},
                        {3 * hidden_size, hidden_size}, {3 * hidden_size}},
        };
        // if (seqAxis == 0) {
        //     // swap batch and seqLengths
        //     std::swap(inputShapes[0][0], inputShapes[0][1]);
        // }
        ov::ParameterVector outerParams{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[0])),
                                            std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[1]))};

        // 1. Create TensorIterator body.
        inputShapes[0][seqAxis] = 1; // sliced dimension
        ov::ParameterVector bodyParams{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[0])),
                                        std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[1]))};

        std::vector<ngraph::Shape> WRB = {inputShapes[2], inputShapes[3], inputShapes[4]};
        auto squeeze = std::make_shared<ngraph::opset5::Squeeze>(bodyParams[0], axis);
        ngraph::OutputVector out_vector = {squeeze, bodyParams[1]};
        auto gru_cell = ngraph::builder::makeGRU(out_vector, WRB, hidden_size, {"sigmoid", "tanh"},
                                                    {}, {}, clip, false);
        auto unsqueeze = std::make_shared<ngraph::opset5::Unsqueeze>(gru_cell->output(0), axis);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gru_cell->output(0)),
                                        std::make_shared<ngraph::opset1::Result>(unsqueeze)};
        auto body = std::make_shared<ngraph::Function>(results, bodyParams, "gru_cell");
        tensorIterator->set_function(body);

        // 2. Set PortMap
        // if (direction == ngraph::op::RecurrentSequenceDirection::FORWARD) {
            // tensorIterator->set_sliced_input(bodyParams[0], outerParams[0], 0, 1, 1, -1, seqAxis);
            // tensorIterator->get_concatenated_slices(results[1], 0, 1, 1, -1, seqAxis);
        // } else if (direction == ngraph::op::RecurrentSequenceDirection::REVERSE) {
            tensorIterator->set_sliced_input(bodyParams[0], outerParams[0], -1, -1, 1, 0, seqAxis);
            tensorIterator->get_concatenated_slices(results[1], -1, -1, 1, 0, seqAxis);
        // } else {
        //     NGRAPH_CHECK(false, "Bidirectional case is not supported.");
        // }

        tensorIterator->set_merged_input(bodyParams[1], outerParams[1], results[0]);
        tensorIterator->get_iter_value(results[0]);

        // 3. Outer function
        // function = std::make_shared<ngraph::Function>(ngraph::OutputVector{tensorIterator->output(0), tensorIterator->output(1)}, outerParams);
        return std::make_shared<ov::Model>(ov::OutputVector{tensorIterator->output(0), tensorIterator->output(1)}, outerParams);
    }

    static void checkContext(const CudaGraphContext& cudaGraphContext) {
        // AddMul network should have a single CUDA Graph
        // EXPECT_EQ(cudaGraphContext.get_graphs_count(), 1);
    }

    static void checkSubGraph(const SubGraph& subGraph) {
        // Original SubGraph for AddMul network should be CUDA Graph compatible
        // EXPECT_EQ(subGraph.GetCudaGraphCompatibility(), CudaGraphCompatibility::FULL);
    }

    static std::vector<std::vector<CalcType>> calcRefs(
        std::shared_ptr<ov::Model> model,
        const std::vector<std::shared_ptr<ov::Tensor>>& inputs) {
    //     EXPECT_EQ(inputTensors.size(), INPUTS_COUNT);
    //     const auto size = inputTensors[0]->get_size();
    //     std::vector<std::vector<CalcType>> result{std::vector<CalcType>(size)};
    //     std::array<const CalcType*, INPUTS_COUNT> inputs;
    //     for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
    //         inputs[i] = getConstPtr(*inputTensors[i]);
    //     }
    //     EXPECT_EQ(result.size(), 1);
    //     auto& output = result[0];
    //     for (std::size_t i = 0; i < size; ++i) {
    //         output[i] = (inputs[0][i] + inputs[1][i]) * ((inputs[2][i] + inputs[3][i]));
    //     }
    //     return result;

        // ConvertRefsParams();
        // functionRefs->validate_nodes_and_infer_types();
        auto refModel = model->clone();

        auto referenceInputs = std::vector<std::vector<uint8_t>>(inputs.size());
        // auto refInputsTypes = std::vector<ngraph::element::Type>(inputs.size());
        auto refInputsTypes = std::vector<ov::element::Type>(inputs.size());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            // const auto inputSize = input->byteSize();
            const auto inputSize = input->get_byte_size();

            auto& referenceInput = referenceInputs[i];
            referenceInput.resize(inputSize);

            // TODO: get rid of deprecated MemoryBlob
            // auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(input);
            // // IE_ASSERT(memory);
            // OPENVINO_ASSERT(memory);
            // const auto lockedMemory = memory->wmap();
            // const auto buffer = lockedMemory.as<const std::uint8_t *>();
            const auto* buffer = static_cast<const uint8_t*>(input->data());
            std::copy(buffer, buffer + inputSize, referenceInput.data());

            // refInputsTypes[i] = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(memory->getTensorDesc().getPrecision());
            refInputsTypes[i] = CALC_ELEMENT_TYPE;
        }

        // const auto&& outputsInfo = executableNetwork.GetOutputsInfo();
        // std::vector<ngraph::element::Type_t> convertType;
        // convertType.reserve(outputsInfo.size());
        // for (const auto &output : outputsInfo) {
        //     convertType.push_back(
        //         FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(
        //             output.second->getTensorDesc().getPrecision()));
        // }

        std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> expectedOutputs;
        expectedOutputs = ngraph::helpers::interpreterFunction(refModel, referenceInputs, refInputsTypes);

        std::vector<std::vector<CalcType>> res(expectedOutputs.size());
        for (std::size_t i = 0; i < expectedOutputs.size(); ++i) {
            EXPECT_EQ(expectedOutputs[i].first, CALC_ELEMENT_TYPE);
            const auto& expOut = expectedOutputs[i].second;
            auto& resOut = res[i];
            const auto resOutSize = expOut.size() / sizeof(CalcType);
            resOut.resize(resOutSize);

            const auto* buffer = static_cast<const CalcType*>(static_cast<const void*>(expOut.data()));
            std::copy(buffer, buffer + resOutSize, resOut.data());
        }

        return res;
    }
};

class AddTI {
public:
    static void createNetworkInternal(std::shared_ptr<ov::Model>& model) {
//         ov::element::Type prc = CALC_ELEMENT_TYPE;
//         ov::Shape shape{1, 2, 3, 4};
//         ov::ParameterVector params;
//         for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
//             params.emplace_back(std::make_shared<ov::op::v0::Parameter>(prc, shape));
//         }
//         const auto add0 = ngraph::builder::makeEltwise(params[0], params[1], EltwiseTypes::ADD);
//         const auto add1 = ngraph::builder::makeEltwise(params[2], params[3], EltwiseTypes::ADD);

//         constexpr int64_t axis = CONCAT_AXIS;
//         const auto concat =
//             std::make_shared<ov::op::v0::Concat>(ov::OutputVector{add0, add1}, axis);
//         const auto result = std::make_shared<ov::op::v0::Result>(concat);
//         return std::make_shared<ov::Model>(result, params, "AddConcat");

        constexpr size_t seqLengths = 20;
        // bool should_decompose;
        constexpr size_t batch = 1;
        // size_t hidden_size = 10;
        constexpr size_t inputSize = 10;
        constexpr size_t seqAxis = 1;
        // ngraph::helpers::TensorIteratorBody ti_body;
        constexpr float clip = 0.0;
        // ngraph::op::RecurrentSequenceDirection direction;
        // InferenceEngine::Precision netPrecision;
        ov::element::Type ngPrc = CALC_ELEMENT_TYPE;

        auto tensorIterator = std::make_shared<ov::op::v0::TensorIterator>();
        auto axisConstant = std::make_shared<ov::op::v0::Constant>(ngraph::element::i64, ngraph::Shape{1},
                                                               std::vector<int64_t>{static_cast<int64_t>(seqAxis)});
        // std::vector<std::vector<size_t>> inputShapes = {
        //         {{batch, seqLengths, inputSize}, {batch, hidden_size}, {3 * hidden_size, inputSize},
        //                 {3 * hidden_size, hidden_size}, {3 * hidden_size}},
        // };
        // std::vector<std::vector<size_t>> inputShapes = {{{batch, seqLengths, inputSize}, {batch, 1, inputSize}}};
        std::vector<size_t> outerShape = {{batch, seqLengths, inputSize}};
        std::vector<std::vector<size_t>> bodyShapes;
        for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
            bodyShapes.emplace_back(std::vector<size_t>{batch, 1, inputSize});
        }
        // if (seqAxis == 0) {
        //     // swap batch and seqLengths
        //     std::swap(inputShapes[0][0], inputShapes[0][1]);
        // }
        // ov::ParameterVector outerParams{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[0])),
        //                                     std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[1]))};
        ov::ParameterVector outerParams;
        outerParams.emplace_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{outerShape}));
        for (std::size_t i = 1; i < INPUTS_COUNT; ++i) {
            outerParams.emplace_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{bodyShapes[i]}));
        }

        // 1. Create TensorIterator body.
        // inputShapes[0][seqAxis] = 1; // sliced dimension
        for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
            ASSERT_EQ(outerShape.size(), bodyShapes[i].size());
            for (std::size_t j = 0; j < bodyShapes[i].size(); ++j) {
                if (j == seqAxis) {
                    ASSERT_EQ(bodyShapes[i][j], 1);
                    ASSERT_EQ(outerShape[j], seqLengths);
                } else {
                    ASSERT_EQ(bodyShapes[i][j], outerShape[j]);
                }
            }
        }
        // ov::ParameterVector bodyParams{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[0])),
        //                                 std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[1]))};
        ov::ParameterVector bodyParams;
        for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
            bodyParams.emplace_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{bodyShapes[i]}));
        }


        // std::vector<ngraph::Shape> WRB = {inputShapes[2], inputShapes[3], inputShapes[4]};
        // ngraph::OutputVector out_vector = {squeeze, bodyParams[1]};
        // auto gru_cell = ngraph::builder::makeGRU(out_vector, WRB, hidden_size, {"sigmoid", "tanh"},
        //                                             {}, {}, clip, false);
        // ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gru_cell->output(0)),
        //                                 std::make_shared<ngraph::opset1::Result>(unsqueeze)};
        // auto body = std::make_shared<ngraph::Function>(results, bodyParams, "gru_cell");

        // auto splitAxisConstant = std::make_shared<ov::op::v0::Constant>(ngraph::element::i64, ov::Shape{1}, CONCAT_AXIS);
        // const auto split =
        //     std::make_shared<ov::op::v1::Split>(concat->output(0), splitAxisConstant, 2);

        // auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(add0->output(0), axis);
        // ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(add0->output(0)),
        //                                 std::make_shared<ov::op::v0::Result>(unsqueeze)};

        // auto squeeze = std::make_shared<ov::op::v0::Squeeze>(bodyParams[0], axisConstant);
        // // TODO: remove ngraph::
        // const auto add0 = ngraph::builder::makeEltwise(squeeze, bodyParams[1], EltwiseTypes::ADD);

        // const auto add1 = ngraph::builder::makeEltwise(bodyParams[2], bodyParams[3], EltwiseTypes::ADD);
        // const auto concat =
        //     std::make_shared<ov::op::v0::Concat>(ov::OutputVector{add0, add1}, CONCAT_AXIS);

        // const auto split = ngraph::builder::makeSplit(concat->output(0), CALC_ELEMENT_TYPE, 2, CONCAT_AXIS);
        // const auto add2 = ngraph::builder::makeEltwise(split->output(0), split->output(1), EltwiseTypes::ADD);

        // auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(add2->output(0), axisConstant);
        // ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(add2->output(0)),
        //                                 std::make_shared<ov::op::v0::Result>(unsqueeze)};

        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(bodyParams[0], axisConstant);
        // TODO: remove ngraph::
        // const auto split = ngraph::builder::makeSplit(squeeze, CALC_ELEMENT_TYPE, 2, 1);
        // const auto concat =
        //     std::make_shared<ov::op::v0::Concat>(ov::OutputVector{split->output(0), split->output(1)}, 1);
        const auto add0 = ngraph::builder::makeEltwise(squeeze->output(0), bodyParams[1], EltwiseTypes::ADD);

        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(add0->output(0), axisConstant);
        ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(add0->output(0)),
                                        std::make_shared<ov::op::v0::Result>(unsqueeze)};

        auto body = std::make_shared<ngraph::Function>(results, bodyParams, "AddConcat");
        tensorIterator->set_function(body);

        // 2. Set PortMap
        // if (direction == ngraph::op::RecurrentSequenceDirection::FORWARD) {
            // tensorIterator->set_sliced_input(bodyParams[0], outerParams[0], 0, 1, 1, -1, seqAxis);
            // tensorIterator->get_concatenated_slices(results[1], 0, 1, 1, -1, seqAxis);
        // } else if (direction == ngraph::op::RecurrentSequenceDirection::REVERSE) {
            tensorIterator->set_sliced_input(bodyParams[0], outerParams[0], -1, -1, 1, 0, seqAxis);
            tensorIterator->get_concatenated_slices(results[1], -1, -1, 1, 0, seqAxis);
        // } else {
        //     NGRAPH_CHECK(false, "Bidirectional case is not supported.");
        // }

        tensorIterator->set_merged_input(bodyParams[1], outerParams[1], results[0]);
        // tensorIterator->set_merged_input(bodyParams[2], outerParams[2], results[0]);
        // tensorIterator->set_merged_input(bodyParams[3], outerParams[3], results[0]);
        tensorIterator->get_iter_value(results[0]);

        // 3. Outer function
        // function = std::make_shared<ngraph::Function>(ngraph::OutputVector{tensorIterator->output(0), tensorIterator->output(1)}, outerParams);
        // return std::make_shared<ov::Model>(ov::OutputVector{tensorIterator->output(0), tensorIterator->output(1)}, outerParams);
        model = std::make_shared<ov::Model>(ov::OutputVector{tensorIterator->output(0), tensorIterator->output(1)}, outerParams);
    }

    static std::shared_ptr<ov::Model> createNetwork() {
        std::shared_ptr<ov::Model> model;
        createNetworkInternal(model);
        return model;
    }

    static void checkContext(const CudaGraphContext& cudaGraphContext) {
        // AddMul network should have a single CUDA Graph
        // EXPECT_EQ(cudaGraphContext.get_graphs_count(), 1);
    }

    static void checkSubGraph(const SubGraph& subGraph) {
        // Original SubGraph for AddMul network should be CUDA Graph compatible
        // EXPECT_EQ(subGraph.GetCudaGraphCompatibility(), CudaGraphCompatibility::FULL);
    }

    static std::vector<std::vector<CalcType>> calcRefs(
        std::shared_ptr<ov::Model> model,
        const std::vector<std::shared_ptr<ov::Tensor>>& inputs) {
    //     EXPECT_EQ(inputTensors.size(), INPUTS_COUNT);
    //     const auto size = inputTensors[0]->get_size();
    //     std::vector<std::vector<CalcType>> result{std::vector<CalcType>(size)};
    //     std::array<const CalcType*, INPUTS_COUNT> inputs;
    //     for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
    //         inputs[i] = getConstPtr(*inputTensors[i]);
    //     }
    //     EXPECT_EQ(result.size(), 1);
    //     auto& output = result[0];
    //     for (std::size_t i = 0; i < size; ++i) {
    //         output[i] = (inputs[0][i] + inputs[1][i]) * ((inputs[2][i] + inputs[3][i]));
    //     }
    //     return result;

        // ConvertRefsParams();
        // functionRefs->validate_nodes_and_infer_types();
        auto refModel = model->clone();

        auto referenceInputs = std::vector<std::vector<uint8_t>>(inputs.size());
        // auto refInputsTypes = std::vector<ngraph::element::Type>(inputs.size());
        auto refInputsTypes = std::vector<ov::element::Type>(inputs.size());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            // const auto inputSize = input->byteSize();
            const auto inputSize = input->get_byte_size();

            auto& referenceInput = referenceInputs[i];
            referenceInput.resize(inputSize);

            // TODO: get rid of deprecated MemoryBlob
            // auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(input);
            // // IE_ASSERT(memory);
            // OPENVINO_ASSERT(memory);
            // const auto lockedMemory = memory->wmap();
            // const auto buffer = lockedMemory.as<const std::uint8_t *>();
            const auto* buffer = static_cast<const uint8_t*>(input->data());
            std::copy(buffer, buffer + inputSize, referenceInput.data());

            // refInputsTypes[i] = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(memory->getTensorDesc().getPrecision());
            refInputsTypes[i] = CALC_ELEMENT_TYPE;
        }

        // const auto&& outputsInfo = executableNetwork.GetOutputsInfo();
        // std::vector<ngraph::element::Type_t> convertType;
        // convertType.reserve(outputsInfo.size());
        // for (const auto &output : outputsInfo) {
        //     convertType.push_back(
        //         FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(
        //             output.second->getTensorDesc().getPrecision()));
        // }

        std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> expectedOutputs;
        expectedOutputs = ngraph::helpers::interpreterFunction(refModel, referenceInputs, refInputsTypes);

        std::vector<std::vector<CalcType>> res(expectedOutputs.size());
        for (std::size_t i = 0; i < expectedOutputs.size(); ++i) {
            EXPECT_EQ(expectedOutputs[i].first, CALC_ELEMENT_TYPE);
            const auto& expOut = expectedOutputs[i].second;
            auto& resOut = res[i];
            const auto resOutSize = expOut.size() / sizeof(CalcType);
            resOut.resize(resOutSize);

            const auto* buffer = static_cast<const CalcType*>(static_cast<const void*>(expOut.data()));
            std::copy(buffer, buffer + resOutSize, resOut.data());
        }

        return res;
    }
};

class SplitConcatAddTI {
public:
    static void createNetworkInternal(std::shared_ptr<ov::Model>& model) {
//         ov::element::Type prc = CALC_ELEMENT_TYPE;
//         ov::Shape shape{1, 2, 3, 4};
//         ov::ParameterVector params;
//         for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
//             params.emplace_back(std::make_shared<ov::op::v0::Parameter>(prc, shape));
//         }
//         const auto add0 = ngraph::builder::makeEltwise(params[0], params[1], EltwiseTypes::ADD);
//         const auto add1 = ngraph::builder::makeEltwise(params[2], params[3], EltwiseTypes::ADD);

//         constexpr int64_t axis = CONCAT_AXIS;
//         const auto concat =
//             std::make_shared<ov::op::v0::Concat>(ov::OutputVector{add0, add1}, axis);
//         const auto result = std::make_shared<ov::op::v0::Result>(concat);
//         return std::make_shared<ov::Model>(result, params, "AddConcat");

        constexpr size_t seqLengths = 20;
        // bool should_decompose;
        constexpr size_t batch = 1;
        // size_t hidden_size = 10;
        constexpr size_t inputSize = 10;
        constexpr size_t seqAxis = 1;
        // ngraph::helpers::TensorIteratorBody ti_body;
        constexpr float clip = 0.0;
        // ngraph::op::RecurrentSequenceDirection direction;
        // InferenceEngine::Precision netPrecision;
        ov::element::Type ngPrc = CALC_ELEMENT_TYPE;

        auto tensorIterator = std::make_shared<ov::op::v0::TensorIterator>();
        auto axisConstant = std::make_shared<ov::op::v0::Constant>(ngraph::element::i64, ngraph::Shape{1},
                                                               std::vector<int64_t>{static_cast<int64_t>(seqAxis)});
        // std::vector<std::vector<size_t>> inputShapes = {
        //         {{batch, seqLengths, inputSize}, {batch, hidden_size}, {3 * hidden_size, inputSize},
        //                 {3 * hidden_size, hidden_size}, {3 * hidden_size}},
        // };
        // std::vector<std::vector<size_t>> inputShapes = {{{batch, seqLengths, inputSize}, {batch, 1, inputSize}}};
        std::vector<size_t> outerShape = {{batch, seqLengths, inputSize}};
        std::vector<std::vector<size_t>> bodyShapes;
        for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
            bodyShapes.emplace_back(std::vector<size_t>{batch, 1, inputSize});
        }
        // if (seqAxis == 0) {
        //     // swap batch and seqLengths
        //     std::swap(inputShapes[0][0], inputShapes[0][1]);
        // }
        // ov::ParameterVector outerParams{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[0])),
        //                                     std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[1]))};
        ov::ParameterVector outerParams;
        outerParams.emplace_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{outerShape}));
        for (std::size_t i = 1; i < INPUTS_COUNT; ++i) {
            outerParams.emplace_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{bodyShapes[i]}));
        }

        // 1. Create TensorIterator body.
        // inputShapes[0][seqAxis] = 1; // sliced dimension
        for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
            ASSERT_EQ(outerShape.size(), bodyShapes[i].size());
            for (std::size_t j = 0; j < bodyShapes[i].size(); ++j) {
                if (j == seqAxis) {
                    ASSERT_EQ(bodyShapes[i][j], 1);
                    ASSERT_EQ(outerShape[j], seqLengths);
                } else {
                    ASSERT_EQ(bodyShapes[i][j], outerShape[j]);
                }
            }
        }
        // ov::ParameterVector bodyParams{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[0])),
        //                                 std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[1]))};
        ov::ParameterVector bodyParams;
        for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
            bodyParams.emplace_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{bodyShapes[i]}));
        }


        // std::vector<ngraph::Shape> WRB = {inputShapes[2], inputShapes[3], inputShapes[4]};
        // ngraph::OutputVector out_vector = {squeeze, bodyParams[1]};
        // auto gru_cell = ngraph::builder::makeGRU(out_vector, WRB, hidden_size, {"sigmoid", "tanh"},
        //                                             {}, {}, clip, false);
        // ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gru_cell->output(0)),
        //                                 std::make_shared<ngraph::opset1::Result>(unsqueeze)};
        // auto body = std::make_shared<ngraph::Function>(results, bodyParams, "gru_cell");

        // auto splitAxisConstant = std::make_shared<ov::op::v0::Constant>(ngraph::element::i64, ov::Shape{1}, CONCAT_AXIS);
        // const auto split =
        //     std::make_shared<ov::op::v1::Split>(concat->output(0), splitAxisConstant, 2);

        // auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(add0->output(0), axis);
        // ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(add0->output(0)),
        //                                 std::make_shared<ov::op::v0::Result>(unsqueeze)};

        // auto squeeze = std::make_shared<ov::op::v0::Squeeze>(bodyParams[0], axisConstant);
        // // TODO: remove ngraph::
        // const auto add0 = ngraph::builder::makeEltwise(squeeze, bodyParams[1], EltwiseTypes::ADD);

        // const auto add1 = ngraph::builder::makeEltwise(bodyParams[2], bodyParams[3], EltwiseTypes::ADD);
        // const auto concat =
        //     std::make_shared<ov::op::v0::Concat>(ov::OutputVector{add0, add1}, CONCAT_AXIS);

        // const auto split = ngraph::builder::makeSplit(concat->output(0), CALC_ELEMENT_TYPE, 2, CONCAT_AXIS);
        // const auto add2 = ngraph::builder::makeEltwise(split->output(0), split->output(1), EltwiseTypes::ADD);

        // auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(add2->output(0), axisConstant);
        // ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(add2->output(0)),
        //                                 std::make_shared<ov::op::v0::Result>(unsqueeze)};

        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(bodyParams[0], axisConstant);
        // TODO: remove ngraph::
        const auto split = ngraph::builder::makeSplit(squeeze, CALC_ELEMENT_TYPE, 2, 1);
        const auto concat =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{split->output(0), split->output(1)}, 1);
        const auto add0 = ngraph::builder::makeEltwise(concat->output(0), bodyParams[1], EltwiseTypes::ADD);

        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(add0->output(0), axisConstant);
        ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(add0->output(0)),
                                        std::make_shared<ov::op::v0::Result>(unsqueeze)};

        auto body = std::make_shared<ngraph::Function>(results, bodyParams, "AddConcat");
        tensorIterator->set_function(body);

        // 2. Set PortMap
        // if (direction == ngraph::op::RecurrentSequenceDirection::FORWARD) {
            // tensorIterator->set_sliced_input(bodyParams[0], outerParams[0], 0, 1, 1, -1, seqAxis);
            // tensorIterator->get_concatenated_slices(results[1], 0, 1, 1, -1, seqAxis);
        // } else if (direction == ngraph::op::RecurrentSequenceDirection::REVERSE) {
            tensorIterator->set_sliced_input(bodyParams[0], outerParams[0], -1, -1, 1, 0, seqAxis);
            tensorIterator->get_concatenated_slices(results[1], -1, -1, 1, 0, seqAxis);
        // } else {
        //     NGRAPH_CHECK(false, "Bidirectional case is not supported.");
        // }

        tensorIterator->set_merged_input(bodyParams[1], outerParams[1], results[0]);
        // tensorIterator->set_merged_input(bodyParams[2], outerParams[2], results[0]);
        // tensorIterator->set_merged_input(bodyParams[3], outerParams[3], results[0]);
        tensorIterator->get_iter_value(results[0]);

        // 3. Outer function
        // function = std::make_shared<ngraph::Function>(ngraph::OutputVector{tensorIterator->output(0), tensorIterator->output(1)}, outerParams);
        // return std::make_shared<ov::Model>(ov::OutputVector{tensorIterator->output(0), tensorIterator->output(1)}, outerParams);
        model = std::make_shared<ov::Model>(ov::OutputVector{tensorIterator->output(0), tensorIterator->output(1)}, outerParams);
    }

    static std::shared_ptr<ov::Model> createNetwork() {
        std::shared_ptr<ov::Model> model;
        createNetworkInternal(model);
        return model;
    }

    static void checkContext(const CudaGraphContext& cudaGraphContext) {
        // AddMul network should have a single CUDA Graph
        // EXPECT_EQ(cudaGraphContext.get_graphs_count(), 1);
    }

    static void checkSubGraph(const SubGraph& subGraph) {
        // Original SubGraph for AddMul network should be CUDA Graph compatible
        // EXPECT_EQ(subGraph.GetCudaGraphCompatibility(), CudaGraphCompatibility::FULL);
    }

    static std::vector<std::vector<CalcType>> calcRefs(
        std::shared_ptr<ov::Model> model,
        const std::vector<std::shared_ptr<ov::Tensor>>& inputs) {
    //     EXPECT_EQ(inputTensors.size(), INPUTS_COUNT);
    //     const auto size = inputTensors[0]->get_size();
    //     std::vector<std::vector<CalcType>> result{std::vector<CalcType>(size)};
    //     std::array<const CalcType*, INPUTS_COUNT> inputs;
    //     for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
    //         inputs[i] = getConstPtr(*inputTensors[i]);
    //     }
    //     EXPECT_EQ(result.size(), 1);
    //     auto& output = result[0];
    //     for (std::size_t i = 0; i < size; ++i) {
    //         output[i] = (inputs[0][i] + inputs[1][i]) * ((inputs[2][i] + inputs[3][i]));
    //     }
    //     return result;

        // ConvertRefsParams();
        // functionRefs->validate_nodes_and_infer_types();
        auto refModel = model->clone();

        auto referenceInputs = std::vector<std::vector<uint8_t>>(inputs.size());
        // auto refInputsTypes = std::vector<ngraph::element::Type>(inputs.size());
        auto refInputsTypes = std::vector<ov::element::Type>(inputs.size());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            // const auto inputSize = input->byteSize();
            const auto inputSize = input->get_byte_size();

            auto& referenceInput = referenceInputs[i];
            referenceInput.resize(inputSize);

            // TODO: get rid of deprecated MemoryBlob
            // auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(input);
            // // IE_ASSERT(memory);
            // OPENVINO_ASSERT(memory);
            // const auto lockedMemory = memory->wmap();
            // const auto buffer = lockedMemory.as<const std::uint8_t *>();
            const auto* buffer = static_cast<const uint8_t*>(input->data());
            std::copy(buffer, buffer + inputSize, referenceInput.data());

            // refInputsTypes[i] = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(memory->getTensorDesc().getPrecision());
            refInputsTypes[i] = CALC_ELEMENT_TYPE;
        }

        // const auto&& outputsInfo = executableNetwork.GetOutputsInfo();
        // std::vector<ngraph::element::Type_t> convertType;
        // convertType.reserve(outputsInfo.size());
        // for (const auto &output : outputsInfo) {
        //     convertType.push_back(
        //         FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(
        //             output.second->getTensorDesc().getPrecision()));
        // }

        std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> expectedOutputs;
        expectedOutputs = ngraph::helpers::interpreterFunction(refModel, referenceInputs, refInputsTypes);

        std::vector<std::vector<CalcType>> res(expectedOutputs.size());
        for (std::size_t i = 0; i < expectedOutputs.size(); ++i) {
            EXPECT_EQ(expectedOutputs[i].first, CALC_ELEMENT_TYPE);
            const auto& expOut = expectedOutputs[i].second;
            auto& resOut = res[i];
            const auto resOutSize = expOut.size() / sizeof(CalcType);
            resOut.resize(resOutSize);

            const auto* buffer = static_cast<const CalcType*>(static_cast<const void*>(expOut.data()));
            std::copy(buffer, buffer + resOutSize, resOut.data());
        }

        return res;
    }
};

// class NestedTI {
// public:
//     static void createNetworkInternal(std::shared_ptr<ov::Model>& model) {
//         constexpr size_t seqLengths = 20;
//         // bool should_decompose;
//         constexpr size_t batch = 1;
//         // size_t hidden_size = 10;
//         constexpr size_t inputSize = 10;
//         constexpr size_t seqAxis = 1;
//         // ngraph::helpers::TensorIteratorBody ti_body;
//         constexpr float clip = 0.0;
//         // ngraph::op::RecurrentSequenceDirection direction;
//         // InferenceEngine::Precision netPrecision;
//         ov::element::Type ngPrc = CALC_ELEMENT_TYPE;

//         auto tensorIterator = std::make_shared<ov::op::v0::TensorIterator>();
//         auto axisConstant = std::make_shared<ov::op::v0::Constant>(ngraph::element::i64, ngraph::Shape{1},
//                                                                std::vector<int64_t>{static_cast<int64_t>(seqAxis)});
//         std::vector<size_t> outerShape = {{batch, 1, seqLengths, inputSize}};
//         std::vector<std::vector<size_t>> bodyShapes = {{batch, 1, inputSize}, {batch, 1, inputSize}};
//         // for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
//         //     bodyShapes.emplace_back(std::vector<size_t>{batch, 1, inputSize});
//         // }
//         // if (seqAxis == 0) {
//         //     // swap batch and seqLengths
//         //     std::swap(inputShapes[0][0], inputShapes[0][1]);
//         // }


//         // ov::ParameterVector outerParams = {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{outerShape}),
//         //                                    std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{bodyShapes[1]})};


//         // outerParams.emplace_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{outerShape}));
//         // for (std::size_t i = 1; i < INPUTS_COUNT; ++i) {
//         //     outerParams.emplace_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{bodyShapes[i]}));
//         // }

//         // 1. Create TensorIterator body.
//         // inputShapes[0][seqAxis] = 1; // sliced dimension
//         // for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
//         //     ASSERT_EQ(outerShape.size(), bodyShapes[i].size());
//         //     for (std::size_t j = 0; j < bodyShapes[i].size(); ++j) {
//         //         if (j == seqAxis) {
//         //             ASSERT_EQ(bodyShapes[i][j], 1);
//         //             ASSERT_EQ(outerShape[j], seqLengths);
//         //         } else {
//         //             ASSERT_EQ(bodyShapes[i][j], outerShape[j]);
//         //         }
//         //     }
//         // }
//         // ov::ParameterVector bodyParams{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[0])),
//         //                                 std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[1]))};
//         ov::ParameterVector bodyParams = {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{bodyShapes[0]}),
//                                           std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{bodyShapes[1]})};
//         // for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
//         //     bodyParams.emplace_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{bodyShapes[i]}));
//         // }

//         auto squeeze = std::make_shared<ov::op::v0::Squeeze>(bodyParams[0], axisConstant);
//         // TODO: remove ngraph::
//         const auto split = ngraph::builder::makeSplit(squeeze, CALC_ELEMENT_TYPE, 2, 1);
//         const auto concat =
//             std::make_shared<ov::op::v0::Concat>(ov::OutputVector{split->output(0), split->output(1)}, 1);
//         const auto add0 = ngraph::builder::makeEltwise(concat->output(0), bodyParams[1], EltwiseTypes::ADD);

//         auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(add0->output(0), axisConstant);
//         ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(add0->output(0)),
//                                         std::make_shared<ov::op::v0::Result>(unsqueeze)};

//         auto body = std::make_shared<ngraph::Function>(results, bodyParams, "AddConcat");
//         tensorIterator->set_function(body);

//         // 2. Set PortMap
//         // if (direction == ngraph::op::RecurrentSequenceDirection::FORWARD) {
//             // tensorIterator->set_sliced_input(bodyParams[0], outerParams[0], 0, 1, 1, -1, seqAxis);
//             // tensorIterator->get_concatenated_slices(results[1], 0, 1, 1, -1, seqAxis);
//         // } else if (direction == ngraph::op::RecurrentSequenceDirection::REVERSE) {

//         std::vector<size_t> outerShape2 = {{batch, seqLengths, seqLengths, inputSize}};
//         ov::ParameterVector outerParams2 = {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{outerShape2}),
//                                             std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{outerShape})};
//         ov::ParameterVector bodyParams2 = {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{outerShape}),
//                                            std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{outerShape})};

//             auto squeeze2 = std::make_shared<ov::op::v0::Squeeze>(bodyParams2[0], axisConstant);
//             const auto add2 = ngraph::builder::makeEltwise(squeeze2->output(0), bodyParams2[1], EltwiseTypes::ADD);
//             // auto unsqueeze2 = std::make_shared<ov::op::v0::Unsqueeze>(add0->output(0), axisConstant);
//             tensorIterator->set_sliced_input(bodyParams[0], add2->output(0), -1, -1, 1, 0, seqAxis);
//             tensorIterator->get_concatenated_slices(results[1], -1, -1, 1, 0, seqAxis);
//         // } else {
//         //     NGRAPH_CHECK(false, "Bidirectional case is not supported.");
//         // }

//         tensorIterator->set_merged_input(bodyParams[1], squeeze2, results[0]);
//         tensorIterator->get_iter_value(results[0]);

//         // auto unsqueeze2 = std::make_shared<ov::op::v0::Unsqueeze>(tensorIterator->output(0), axisConstant);


//         ngraph::ResultVector results2 = {std::make_shared<ov::op::v0::Result>(tensorIterator->output(0)),
//                                          std::make_shared<ov::op::v0::Result>(tensorIterator->output(1))};

//         auto body2 = std::make_shared<ngraph::Function>(results2, bodyParams2, "OuterTI");

//         auto tensorIterator2 = std::make_shared<ov::op::v0::TensorIterator>();
//         tensorIterator2->set_function(body2);

//         tensorIterator2->set_sliced_input(bodyParams2[0], outerParams2[0], -1, -1, 1, 0, seqAxis);
//         tensorIterator2->get_concatenated_slices(results2[1], -1, -1, 1, 0, seqAxis);

//         tensorIterator2->set_merged_input(bodyParams2[1], outerParams2[1], results2[0]);
//         tensorIterator2->get_iter_value(results2[0]);

//         model = std::make_shared<ov::Model>(ov::OutputVector{tensorIterator2->output(0), tensorIterator2->output(1)}, outerParams2);
// }

//     static std::shared_ptr<ov::Model> createNetwork() {
//         std::shared_ptr<ov::Model> model;
//         createNetworkInternal(model);
//         return model;
//     }

//     static void checkContext(const CudaGraphContext& cudaGraphContext) {
//         // AddMul network should have a single CUDA Graph
//         // EXPECT_EQ(cudaGraphContext.get_graphs_count(), 1);
//     }

//     static void checkSubGraph(const SubGraph& subGraph) {
//         // Original SubGraph for AddMul network should be CUDA Graph compatible
//         // EXPECT_EQ(subGraph.GetCudaGraphCompatibility(), CudaGraphCompatibility::FULL);
//     }

//     static std::vector<std::vector<CalcType>> calcRefs(
//         std::shared_ptr<ov::Model> model,
//         const std::vector<std::shared_ptr<ov::Tensor>>& inputs) {
//     //     EXPECT_EQ(inputTensors.size(), INPUTS_COUNT);
//     //     const auto size = inputTensors[0]->get_size();
//     //     std::vector<std::vector<CalcType>> result{std::vector<CalcType>(size)};
//     //     std::array<const CalcType*, INPUTS_COUNT> inputs;
//     //     for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
//     //         inputs[i] = getConstPtr(*inputTensors[i]);
//     //     }
//     //     EXPECT_EQ(result.size(), 1);
//     //     auto& output = result[0];
//     //     for (std::size_t i = 0; i < size; ++i) {
//     //         output[i] = (inputs[0][i] + inputs[1][i]) * ((inputs[2][i] + inputs[3][i]));
//     //     }
//     //     return result;

//         // ConvertRefsParams();
//         // functionRefs->validate_nodes_and_infer_types();
//         auto refModel = model->clone();

//         auto referenceInputs = std::vector<std::vector<uint8_t>>(inputs.size());
//         // auto refInputsTypes = std::vector<ngraph::element::Type>(inputs.size());
//         auto refInputsTypes = std::vector<ov::element::Type>(inputs.size());
//         for (std::size_t i = 0; i < inputs.size(); ++i) {
//             const auto& input = inputs[i];
//             // const auto inputSize = input->byteSize();
//             const auto inputSize = input->get_byte_size();

//             auto& referenceInput = referenceInputs[i];
//             referenceInput.resize(inputSize);

//             // TODO: get rid of deprecated MemoryBlob
//             // auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(input);
//             // // IE_ASSERT(memory);
//             // OPENVINO_ASSERT(memory);
//             // const auto lockedMemory = memory->wmap();
//             // const auto buffer = lockedMemory.as<const std::uint8_t *>();
//             const auto* buffer = static_cast<const uint8_t*>(input->data());
//             std::copy(buffer, buffer + inputSize, referenceInput.data());

//             // refInputsTypes[i] = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(memory->getTensorDesc().getPrecision());
//             refInputsTypes[i] = CALC_ELEMENT_TYPE;
//         }

//         // const auto&& outputsInfo = executableNetwork.GetOutputsInfo();
//         // std::vector<ngraph::element::Type_t> convertType;
//         // convertType.reserve(outputsInfo.size());
//         // for (const auto &output : outputsInfo) {
//         //     convertType.push_back(
//         //         FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(
//         //             output.second->getTensorDesc().getPrecision()));
//         // }

//         std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> expectedOutputs;
//         expectedOutputs = ngraph::helpers::interpreterFunction(refModel, referenceInputs, refInputsTypes);

//         std::vector<std::vector<CalcType>> res(expectedOutputs.size());
//         for (std::size_t i = 0; i < expectedOutputs.size(); ++i) {
//             EXPECT_EQ(expectedOutputs[i].first, CALC_ELEMENT_TYPE);
//             const auto& expOut = expectedOutputs[i].second;
//             auto& resOut = res[i];
//             const auto resOutSize = expOut.size() / sizeof(CalcType);
//             resOut.resize(resOutSize);

//             const auto* buffer = static_cast<const CalcType*>(static_cast<const void*>(expOut.data()));
//             std::copy(buffer, buffer + resOutSize, resOut.data());
//         }

//         return res;
//     }
// };



template <typename Network>
class CudaMultiGraphTest : public Test {
protected:
    static std::map<std::string, std::size_t> populateInputIndices(std::shared_ptr<ov::Model> model) {
        std::map<std::string, std::size_t> inputIndices;
        for (const auto& parameter : model->get_parameters()) {
            const auto& parameter_index = model->get_parameter_index(parameter);
            inputIndices.emplace(ParameterOp::GetInputTensorName(*parameter), parameter_index);
        }
        return inputIndices;
    }

    static std::map<std::string, std::size_t> populateOutputIndices(std::shared_ptr<ov::Model> model) {
        std::map<std::string, std::size_t> outputIndices;
        for (auto& result : model->get_results()) {
            const auto& result_index = model->get_result_index(result->input_value(0));
            for (const auto& outputName : ResultOp::GetOutputTensorName(*result)) {
                outputIndices.emplace(outputName, result_index);
            }
        }
        return outputIndices;
    }

    static std::vector<std::shared_ptr<ov::Tensor>> populateTensors(const std::vector<ov::Output<ov::Node>>& nodes) {
        std::vector<std::shared_ptr<ov::Tensor>> result;
        for (const auto& node : nodes) {
            result.push_back(std::make_shared<ov::Tensor>(node.get_element_type(), node.get_shape()));
        }
        return result;
    }

    void generateInputs() {
        for (auto& input : inputTensors_) {
            generateInput(*input, TO, FROM, currentSeed_);
            ++currentSeed_;
        }
    }

    void updateContext() { runner_.UpdateContext(*inferRequestContext_, deviceMemBlock_); }

    void checkConditions() {
        Network::checkContext(cudaGraphContext_);
        Network::checkSubGraph(runner_.GetSubGraph());
    }

    void run() { runner_.Run(*inferRequestContext_, deviceMemBlock_); }

    void calcRefs() { refOutputs_ = Network::calcRefs(model_,  inputTensors_); }
    // void calcRefs() {}

    void validate(float threshold = THRESHOLD) {
        const auto size = outputTensors_.size();
        EXPECT_EQ(size, refOutputs_.size());
        for (std::size_t i = 0; i < size; ++i) {
            validateOutput(*outputTensors_[i], refOutputs_[i], THRESHOLD);
        }
    }

    void updateTensors() {
        inputTensors_ = {populateTensors(model_->inputs())};
        outputTensors_ = {populateTensors(model_->outputs())};
        inferRequestContext_ = std::make_unique<InferenceRequestContext>(inputTensors_,
                                                                         inputIndices_,
                                                                         outputTensors_,
                                                                         outputIndices_,
                                                                         threadContext_,
                                                                         cancellationToken_,
                                                                         simpleExecutionDelegator_,
                                                                         cudaGraphContext_,
                                                                         false);
    }

    void runTest() {
        generateInputs();
        updateContext();
        checkConditions();
        run();
        calcRefs();
        validate();

        updateTensors();
        generateInputs();
        updateContext();
        checkConditions();
        run();
        calcRefs();
        validate();
    }

    std::shared_ptr<ov::Model> model_{Network::createNetwork()};
    CreationContext creationContext_{{}, false};
    ThreadContext threadContext_{{}};
    CancellationToken cancellationToken_{};
    CudaGraphContext cudaGraphContext_{};
    // CudaGraphTopologyRunner runner_{creationContext_, model_};
    EagerTopologyRunner runner_{creationContext_, model_};
    SimpleExecutionDelegator simpleExecutionDelegator_{};
    std::vector<std::shared_ptr<ov::Tensor>> inputTensors_{populateTensors(model_->inputs())};
    std::vector<std::shared_ptr<ov::Tensor>> outputTensors_{populateTensors(model_->outputs())};
    std::map<std::string, std::size_t> inputIndices_{populateInputIndices(model_)};
    std::map<std::string, std::size_t> outputIndices_{populateOutputIndices(model_)};
    std::unique_ptr<InferenceRequestContext> inferRequestContext_ =
        std::make_unique<InferenceRequestContext>(inputTensors_,
                                                  inputIndices_,
                                                  outputTensors_,
                                                  outputIndices_,
                                                  threadContext_,
                                                  cancellationToken_,
                                                  simpleExecutionDelegator_,
                                                  cudaGraphContext_,
                                                  false);
    DeviceMemBlock deviceMemBlock_{runner_.GetSubGraph().memoryManager()->mutableTensorsMemoryModel()};

    std::vector<std::vector<CalcType>> refOutputs_;
    int currentSeed_ = SEED;
};

using GRUTIMultiGraphTest = CudaMultiGraphTest<GRUTI>;

TEST_F(GRUTIMultiGraphTest, GRUTIMultiGraphTest) { runTest(); }

using AddTIMultiGraphTest = CudaMultiGraphTest<AddTI>;

TEST_F(AddTIMultiGraphTest, AddTIMultiGraphTest) { runTest(); }

using SplitConcatAddTIMultiGraphTest = CudaMultiGraphTest<SplitConcatAddTI>;

TEST_F(SplitConcatAddTIMultiGraphTest, SplitConcatAddTIMultiGraphTest) { runTest(); }

// using NestedTIMultiGraphTest = CudaMultiGraphTest<NestedTI>;

// TEST_F(NestedTIMultiGraphTest, NestedTIMultiGraphTest) { runTest(); }
