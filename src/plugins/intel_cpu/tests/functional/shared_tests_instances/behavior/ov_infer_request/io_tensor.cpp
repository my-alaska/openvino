// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/io_tensor.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {
        {},
        {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}},
        {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "0"}, {InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, "1"}}
};

const std::vector<ov::AnyMap> emptyConfigs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestIOTensorTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::ValuesIn(configs)),
                        OVInferRequestIOTensorTest::getTestCaseName);

std::vector<ov::element::Type> prcs = {
    ov::element::boolean,
    ov::element::bf16,
    ov::element::f16,
    ov::element::f32,
    ov::element::f64,
    ov::element::i4,
    ov::element::i8,
    ov::element::i16,
    ov::element::i32,
    ov::element::i64,
    ov::element::u1,
    ov::element::u4,
    ov::element::u8,
    ov::element::u16,
    ov::element::u32,
    ov::element::u64,
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestIOTensorSetPrecisionTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(prcs),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::ValuesIn(configs)),
                         OVInferRequestIOTensorSetPrecisionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(
                                 ::testing::ValuesIn(prcs),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::ValuesIn(emptyConfigs)),
                         OVInferRequestCheckTensorPrecision::getTestCaseName);
}  // namespace
