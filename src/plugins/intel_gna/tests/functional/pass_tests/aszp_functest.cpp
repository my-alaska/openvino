// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <string>

#include "../shared_tests_instances/skip_tests_check.hpp"
#include "common_test_utils/test_common.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "transformations/init_node_info.hpp"



using namespace ngraph;
using namespace ngraph::opset11;

namespace LayerTestsDefinitions {

typedef std::tuple<InferenceEngine::SizeVector,  // Kernel size
                   InferenceEngine::SizeVector,  // Strides
                   std::vector<ptrdiff_t>,       // Pad begin
                   std::vector<ptrdiff_t>,       // Pad end
                   InferenceEngine::SizeVector,  // Dilation
                   size_t                        // Num out channels
                   >
    convSpecificParams;

typedef std::tuple<convSpecificParams,                  // Convolution parameters
                   InferenceEngine::Precision,          // Network Precision
                   InferenceEngine::SizeVector          // Input shapes
                   >
    AsymmetricPaddingParams;

class AsymmetricToSymmetricPaddingConvTest : public testing::WithParamInterface<AsymmetricPaddingParams>,
                              virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<AsymmetricPaddingParams> obj) {
        convSpecificParams convParams;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::SizeVector inputShape;
        std::tie(convParams, netPrecision, inputShape) = obj.param;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t numOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, numOutChannels) = convParams;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "K" << CommonTestUtils::vec2str(kernel) << "_";
        result << "S" << CommonTestUtils::vec2str(stride) << "_";
        result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
        result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
        result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
        result << "O=" << numOutChannels << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        return result.str();
    }

protected:
    GnaLayerTestCheck gnaVersionCheck;

    void SetUp() override {
        threshold = 0.015f;
        convSpecificParams convParams;
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        std::tie(convParams, netPrecision, inputShape) =
            this->GetParam();
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t numOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, numOutChannels) = convParams;

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto input = builder::makeParams(ngPrc, {inputShape});
        auto transposeIn = std::make_shared<Transpose>(input[0], op::Constant::create(element::i64, Shape{4}, {0, 3, 1, 2}));
        auto filterSize = std::accumulate(std::begin(kernel), std::end(kernel), 1ull, std::multiplies<size_t>());
        auto filterWeights =
            CommonTestUtils::generate_float_numbers(numOutChannels * inputShape[3] * filterSize, -0.05f, 0.05f);
        auto conv = builder::makeConvolution(transposeIn,
                                             ngPrc,
                                             kernel,
                                             stride,
                                             padBegin,
                                             padEnd,
                                             dilation,
                                             op::PadType::EXPLICIT,
                                             numOutChannels,
                                             false,
                                             filterWeights);
        auto transposeOutOrder = op::Constant::create(element::i64, Shape{4}, {0, 2, 3, 1});
        Output<Node> lastOp = std::make_shared<Transpose>(conv, transposeOutOrder);

        auto result = std::make_shared<Result>(lastOp);
        function = std::make_shared<Function>(ResultVector{result}, ParameterVector{input});
        gnaVersionCheck.SetUp(CommonTestUtils::DEVICE_GNA);
    }
};

TEST_P(AsymmetricToSymmetricPaddingConvTest, CompareWithRefs) {
    Run();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::vector<size_t>> input1DNHWC = {{1, 1, 16, 8}};
const std::vector<std::vector<size_t>> kernels1D = {{1, 2}, {1, 3}, {1, 4}};
const std::vector<std::vector<size_t>> strides1D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins1D = {{0, 2}};
const std::vector<std::vector<ptrdiff_t>> padEnds1D = {{0, 3}};
const std::vector<std::vector<size_t>> dilations1D = {{1, 1}};
const std::vector<size_t> numOutChannels1D = {4};
const std::vector<std::vector<size_t>> biases1D = {{1, 4, 1, 1}};
const std::vector<std::vector<size_t>> transpBiases1D = {{1, 1, 1, 4}};
const std::vector<std::vector<size_t>> maxpool1DPools = {{1, 2}};
const std::vector<std::vector<size_t>> maxpool1DStrides = {{1, 1}};

const std::vector<std::vector<size_t>> input2DNHWC = {{1, 16, 16, 32}};
const std::vector<std::vector<size_t>> kernels2D = {{2, 2}, {4, 1}};
const std::vector<std::vector<size_t>> strides2D = {{1, 1}, {2, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{1, 2}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{3, 1}};
const std::vector<std::vector<size_t>> dilations2D = {{1, 1}};
const std::vector<size_t> numOutChannels2D = {8};
const std::vector<std::vector<size_t>> biases2D = {{1, 8, 1, 1}};
const std::vector<std::vector<size_t>> transpBiases2D = {{1, 1, 1, 8}};
const std::vector<std::vector<size_t>> maxpool2DPools = {{2, 2}};
const std::vector<std::vector<size_t>> maxpool2DStrides = {{2, 1}};

const auto conv1DParams = ::testing::Combine(::testing::ValuesIn(kernels1D),
                                             ::testing::ValuesIn(strides1D),
                                             ::testing::ValuesIn(padBegins1D),
                                             ::testing::ValuesIn(padEnds1D),
                                             ::testing::ValuesIn(dilations1D),
                                             ::testing::ValuesIn(numOutChannels1D));

const auto misc1DParams = ::testing::Combine(::testing::ValuesIn(biases1D),
                                             ::testing::ValuesIn(transpBiases1D),
                                             ::testing::ValuesIn(maxpool1DPools),
                                             ::testing::ValuesIn(maxpool1DStrides));

const auto conv2DParams = ::testing::Combine(::testing::ValuesIn(kernels2D),
                                             ::testing::ValuesIn(strides2D),
                                             ::testing::ValuesIn(padBegins2D),
                                             ::testing::ValuesIn(padEnds2D),
                                             ::testing::ValuesIn(dilations2D),
                                             ::testing::ValuesIn(numOutChannels2D));

const auto misc2DParams = ::testing::Combine(::testing::ValuesIn(biases2D),
                                             ::testing::ValuesIn(transpBiases2D),
                                             ::testing::ValuesIn(maxpool2DPools),
                                             ::testing::ValuesIn(maxpool2DStrides));

INSTANTIATE_TEST_SUITE_P(smoke_1DPaddedToValid,
                         AsymmetricToSymmetricPaddingConvTest,
                         ::testing::Combine(conv1DParams,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(input1DNHWC)),
                         AsymmetricToSymmetricPaddingConvTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_2DPaddedToValid,
                         AsymmetricToSymmetricPaddingConvTest,
                         ::testing::Combine(conv2DParams,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(input2DNHWC)),
                         AsymmetricToSymmetricPaddingConvTest::getTestCaseName);

}  // namespace LayerTestsDefinitions
