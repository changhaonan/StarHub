#version 450 core

flat in int vs_out_vertex_id;
in vec4 vs_out_warp_vertex;
in vec4 vs_out_warp_normal;
flat in vec4 vs_out_color_time;

layout(location = 0) out vec4 vertex_confid_map;
layout(location = 1) out vec4 normal_radius_map;
layout(location = 2) out int index_map;
layout(location = 3) out vec4 color_time_map;


void main() {
    index_map = vs_out_vertex_id;
    vertex_confid_map = vs_out_warp_vertex;
    normal_radius_map = vs_out_warp_normal;
    color_time_map = vs_out_color_time;
}