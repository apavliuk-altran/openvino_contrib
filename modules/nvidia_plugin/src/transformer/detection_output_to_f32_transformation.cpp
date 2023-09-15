// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/cc/pass/itt.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "detection_output_to_f32_transformation.hpp"

// #include <cuda_op_buffers_extractor.hpp>
// #include <exec_graph_info.hpp>
// #include <gsl/span_ext>
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/detection_output.hpp"

using namespace ov::pass::pattern;

namespace ov::nvidia_gpu::pass {

bool detection_output_convert_inputs(Matcher &m) {
    auto d_out = std::dynamic_pointer_cast<ov::op::v0::DetectionOutput>(m.get_match_root());
    if (!d_out) {
        return false;
    }

    const auto& type = d_out->get_element_type();
    // if (type != ov::element::Type_t::bf16 && type != ov::element::Type_t::f16 && type != ov::element::Type_t::f64) {
    //     return false;
    // }

    const auto& inputs = d_out->inputs();
    ov::NodeVector input_nodes;
    for (size_t i = 0; i < d_out->inputs().size(); ++i) {
        const auto& input_type = d_out->input(i).get_element_type();
        if (input_type != type) {
            // return false;
            const auto convert = std::make_shared<ov::op::v0::Convert>(d_out->input_value(i), type);
            ov::copy_runtime_info(d_out, convert);
            input_nodes.emplace_back(convert);
        } else {
            input_nodes.emplace_back(d_out->input_value(i).get_node_shared_ptr());
        }
        // input_nodes.emplace_back(std::make_shared<ov::op::v0::Convert>(input.get_source_output(), ov::element::Type_t::f32));
    }

    const auto inputs_size = input_nodes.size();
    const auto& attrs = d_out->get_attrs();
    std::shared_ptr<ov::op::v0::DetectionOutput> new_d_out{nullptr};
    if (inputs_size == 3) {
        new_d_out = std::make_shared<ov::op::v0::DetectionOutput>(input_nodes[0], input_nodes[1], input_nodes[2], attrs);
    } else if (inputs_size == 5) {
        new_d_out = std::make_shared<ov::op::v0::DetectionOutput>(input_nodes[0], input_nodes[1], input_nodes[2], input_nodes[3], input_nodes[4], attrs);
    } else {
        return false;
    }
    new_d_out->set_friendly_name(d_out->get_friendly_name());

    // const auto out_convert = std::make_shared<ov::op::v0::Convert>(new_d_out, type);

    // ov::copy_runtime_info(d_out, input_nodes);
    // ov::copy_runtime_info(d_out, {new_d_out, out_convert});
    ov::copy_runtime_info(d_out, new_d_out);
    // ov::replace_node(d_out, out_convert);
    ov::replace_node(d_out, new_d_out);

    return true;
}

DetectionOutputToF32Transformation::DetectionOutputToF32Transformation() {
    MATCHER_SCOPE(DetectionOutputToF32Transformation);
    // const auto type_pattern = ov::pass::pattern::type_matches_any({element::bf16, element::f16, element::f64});
    // const auto do3 = wrap_type<ov::op::v0::DetectionOutput>({any_input(), any_input(), any_input()}, type_pattern);
    // const auto do5 = wrap_type<ov::op::v0::DetectionOutput>({any_input(), any_input(), any_input(), any_input(), any_input()}, type_pattern);
    const auto do3 = wrap_type<ov::op::v0::DetectionOutput>({any_input(), any_input(), any_input()});
    // const auto do5 = wrap_type<ov::op::v0::DetectionOutput>({any_input(), any_input(), any_input(), any_input(), any_input()});
    matcher_pass_callback callback = [](Matcher &m) { return detection_output_convert_inputs(m); };

    const auto m3 = std::make_shared<Matcher>(do3, matcher_name);
    register_matcher(m3, callback);

    // auto m5 = std::make_shared<Matcher>(do5, matcher_name);
    // register_matcher(m5, callback);
}

}  // namespace ov::nvidia_gpu::pass
