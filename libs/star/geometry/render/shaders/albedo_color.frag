#version 450 core

in VertexOut {
    vec4 camera_vertex;
    vec4 camera_normal;
    vec4 normalized_rgbd;
    flat int vertex_id;
} fs_in;

out vec4 fragment_color;

void main() {
    fragment_color = vec4(fs_in.normalized_rgbd.xyz, 1.0);
}