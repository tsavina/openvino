// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
OutputVector base_expand(const NodeContext& context, const Output<Node>& x, const Output<Node>& sizes) {
    auto data = x;
    Output<Node> shape = context.mark_node(std::make_shared<v0::Abs>(sizes));
    auto complex = as_type_ptr<ComplexTypeMark>(data.get_node_shared_ptr());
    bool is_complex = complex != nullptr;
    if (is_complex) {
        data = complex->get_input_source_output(0);
        auto two = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {2}));
        if (shape.get_element_type() != element::i32) {
            two = context.mark_node(std::make_shared<v1::ConvertLike>(two, shape));
        }
        shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{shape, two}, 0));
    }
    auto res = context.mark_node(std::make_shared<v3::Broadcast>(data, shape, BroadcastType::BIDIRECTIONAL));
    if (is_complex) {
        res = context.mark_node(std::make_shared<ComplexTypeMark>(res));
    }
    return {res};
};
}  // namespace

OutputVector translate_expand(const NodeContext& context) {
    // aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)
    num_inputs_check(context, 2, 3, true);
    auto x = context.get_input(0);
    auto sizes = get_input_concat_if_list(context, 1);
    // TODO: figure out what implicit means
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(2) || context.const_input<bool>(2) == false,
                                "Unexpected value of implicit for expand operation");
    return base_expand(context, x, sizes);
};

OutputVector translate_expand_as(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(y, element::i32));
    return {context.mark_node(std::make_shared<v3::Broadcast>(x, shape, BroadcastType::BIDIRECTIONAL))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
