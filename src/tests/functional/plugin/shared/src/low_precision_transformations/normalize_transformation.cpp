// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/normalize_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"

#include "ov_lpt_models/normalize_l2.hpp"

namespace LayerTestsDefinitions {

std::string NormalizeL2Transformation::getTestCaseName(const testing::TestParamInfo<NormalizeL2TransformationParams>& obj) {
    auto [netPrecision, shapes, device, axes, fuseMultiply, shift] = obj.param;
    std::ostringstream result;
    result << netPrecision << "_" <<
           shapes.first << "_" <<
           shapes.second << "_" <<
           device << "_" <<
           "_axes" << axes.size() <<
        (fuseMultiply ? "_multiply" : "") <<
        (shift ? "_shift" : "");
    return result.str();
}

void NormalizeL2Transformation::SetUp() {
    auto [precision, shapes, device, axes, fuseMultiply, shift] = this->GetParam();
    targetDevice = device;

    init_input_shapes(shapes.first);

    function = ov::builder::subgraph::NormalizeL2Function::getOriginal(
        precision,
        shapes,
        ov::element::u8,
        axes,
        fuseMultiply,
        shift);
}

TEST_P(NormalizeL2Transformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
