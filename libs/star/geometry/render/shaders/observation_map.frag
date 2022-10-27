#version 450 core

in VertexOut {
    vec4 camera_vertex;
    vec4 camera_normal;
    vec4 normalized_rgbd;
    flat int vertex_id;
} fs_in;

layout(location = 0) out vec4 rgbd_map;
layout(location = 1) out int index_map;

void main() {
    // rgb * 2.0 - 1.0, d is not changed
    rgbd_map = vec4(fs_in.normalized_rgbd.x * 2.0 - 1.0, fs_in.normalized_rgbd.y * 2.0 - 1.0, fs_in.normalized_rgbd.z * 2.0 - 1.0, fs_in.normalized_rgbd.w);
    index_map = fs_in.vertex_id;
}