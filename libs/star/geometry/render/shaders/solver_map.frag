#version 450 core

in vec4 vs_out_reference_vertex;
in vec4 vs_out_reference_normal;
flat in int vs_out_vertex_id;
in vec4 vs_out_normalized_rgbd;

layout(location = 0) out vec4 reference_vertex_map;
layout(location = 1) out vec4 reference_normal_map;
layout(location = 2) out int index_map;
layout(location = 3) out vec4 normalized_rgbd_map;

void main() {
    reference_vertex_map = vs_out_reference_vertex;
    reference_normal_map = vs_out_reference_normal;
    index_map = vs_out_vertex_id;
    normalized_rgbd_map = vs_out_normalized_rgbd;
}