#version 450 core

layout(location = 0) in vec4 reference_vertex_confid;
layout(location = 1) in vec4 reference_normal_radius;
layout(location = 2) in vec4 color_time;

out vec4 vs_out_reference_vertex;
out vec4 vs_out_reference_normal;
flat out int vs_out_vertex_id;
out vec4 vs_out_normalized_rgbd;  // (R/255.0, G/255.0, B/255.0, 1.0/D)

uniform mat4 world2camera;
uniform vec4 intrinsic; // cx, cy, fx, fy
// The width and height are in pixels, the maxdepth is in [m]
// The last element is current time
uniform vec4 width_height_maxdepth_currtime;
uniform vec2 confid_time_threshold; // x is confidence threshold, y is time threshold

// Note: 0.5 shift is needed due to gl_FragCoord is rounding down, not rounding nearest.
// Make sure (-0.5 ~ 0.5) are mapped to 0
vec3 project_point(vec3 p)
{
    return vec3(( (((intrinsic.z * p.x) / p.z) + intrinsic.x + 0.5) - (width_height_maxdepth_currtime.x * 0.5)) / (width_height_maxdepth_currtime.x * 0.5),
                ((((intrinsic.w * p.y) / p.z) + intrinsic.y + 0.5) - (width_height_maxdepth_currtime.y * 0.5)) / (width_height_maxdepth_currtime.y * 0.5),
                p.z / (width_height_maxdepth_currtime.z));
}

void main() {
    vec4 reference_vertex_camera = world2camera * vec4(reference_vertex_confid.xyz, 1.0);
    if(reference_vertex_camera.z > width_height_maxdepth_currtime.z
    || reference_vertex_camera.z <= 0
    || (reference_vertex_confid.w < confid_time_threshold.x && (abs(color_time.z - width_height_maxdepth_currtime.w) >= confid_time_threshold.y))
    ) {
        // Make it outside the screes of [-1 1]
        gl_Position = vec4(1000.0f, 1000.0f, 1000.0f, 1000.0f);
    }
    else {
        gl_Position = vec4(project_point(reference_vertex_camera.xyz), 1.0);

        // Collect information for fragment shader input
        vs_out_reference_vertex = vec4(reference_vertex_confid.xyz, 1.0);
        vs_out_reference_normal = vec4(reference_normal_radius.xyz, 0.0);
        vs_out_vertex_id = gl_VertexID;

        // The normalized color from color_time
        int encoded_color = floatBitsToInt(color_time.x);
        vec3 color;
        color.x = float(((encoded_color & 0x00FF0000) >> 16)) / 127.5f - 1.0f;
        color.y = float(((encoded_color & 0x0000FF00) >>  8)) / 127.5f - 1.0f;
        color.z = float(  encoded_color & 0x000000FF) / 127.5f - 1.0f;
        vs_out_normalized_rgbd = vec4(color.xyz, 1.0 / reference_vertex_camera.z);
    }
}
