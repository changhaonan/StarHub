#include <star/common/logging.h>
#include <fstream>
#include <stdlib.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "global_configs.h"
#include "ConfigParser.h"
#include "Constants.h"

star::ConfigParser::ConfigParser() : m_num_cam(d_max_cam)
{
}

// Access interface
star::ConfigParser &star::ConfigParser::Instance()
{
    static ConfigParser parser;
    return parser;
}

/* Public interface
 */
void star::ConfigParser::ParseConfig(
    const std::string &sys_config_path,
    const std::string &context_config_path,
    const std::string &vis_config_path,
    const std::string &output_path)
{
    // Path setting
    // Context config should be put at the root of data dir
    m_data_prefix = boost::filesystem::path(context_config_path).parent_path().string();
    m_output_path = output_path;

    // System setting
    json sys_config_json;
    std::ifstream sys_input_file(sys_config_path);
    sys_input_file >> sys_config_json;
    sys_input_file.close();
    loadSysConfigFromJson((const void *)&sys_config_json);

    // Camera setting
    json context_config_json;
    std::ifstream context_input_file(context_config_path);
    context_input_file >> context_config_json;
    context_input_file.close();
    loadContextConfigFromJson((const void *)&context_config_json);

    // Vis setting
    json vis_config_json;
    std::ifstream vis_input_file(vis_config_path);
    vis_input_file >> vis_config_json;
    vis_input_file.close();
    loadVisConfigFromJson((const void *)&vis_config_json);
}

void star::ConfigParser::loadSysConfigFromJson(const void *json_ptr)
{
    const auto &config_json = *((const json *)json_ptr);

    // Path-related
    STAR_CHECK(config_json.find("model_path") != config_json.end());
    m_model_path = config_json.at("model_path");
    // Frame-related
    STAR_CHECK(config_json.find("start_frame") != config_json.end());
    STAR_CHECK(config_json.find("num_frames") != config_json.end());
    m_start_frame_idx = config_json.at("start_frame");
    m_num_frames = config_json.at("num_frames");
    m_step_frame = config_json.at("step_frame");
    // Lambda function
    const auto check_and_load = [&](bool &assign_value, const std::string &key, bool default_value) -> void
    {
        // The value is not in the config
        if (config_json.find(key) == config_json.end())
        {
            assign_value = default_value;
            return;
        }
        // In the config
        assign_value = config_json[key];
    };
    const auto check_and_load_string = [&](std::string &assign_value, const std::string &key, std::string default_value) -> void
    {
        // The value is not in the config
        if (config_json.find(key) == config_json.end())
        {
            assign_value = default_value;
            return;
        }
        // In the config
        assign_value = config_json[key];
    };

    // Debug-related
    check_and_load(m_debug_node_external_load, "debug_node_external_load", false);

    // Filter-related
    check_and_load(m_use_gaussian_filter, "use_gaussian_filter", false);

    // Optical-flow-related
    m_opticalflow_suppress_threshold = config_json.at("opticalflow_suppress_threshold");

    // Tsdf-related
    m_tsdf_width = config_json.at("tsdf_width");
    m_tsdf_height = config_json.at("tsdf_height");
    m_tsdf_depth = config_json.at("tsdf_depth");
    m_tsdf_voxel_size_m = config_json.at("tsdf_voxel_size");
    std::vector<float> h_origin_vec = config_json.at("tsdf_origin").get<std::vector<float>>();
    m_tsdf_origin_m = make_float3(
        h_origin_vec[0], h_origin_vec[1], h_origin_vec[2]);

    // Surfel&NodeGraph-related
    m_node_radius = config_json.at("node_radius");
    m_surfel_radius_scale = config_json.at("surfel_radius_scale");
    check_and_load(m_enable_semantic_surfel, "enable_semantic_surfel", false);

    // Object-specified
    m_semantic_label.resize(d_max_num_semantic);
    for (auto i = 0; i < d_max_num_semantic; ++i)
    { // Initialized as all 1.f
        m_dynamic_regulation[i] = 1.f;
        m_semantic_label[i] = 0;
    }
    if (config_json.find("semantic_prior") != config_json.end())
    {
        auto semantic_prior_iter = config_json.find("semantic_prior");
        unsigned sementic_id = 1;
        float max_rigidness = 0.f;
        for (auto &el : (*semantic_prior_iter).items())
        {
            int label = el.value().at("label").get<int>();
            float rigidness = el.value().at("rigidness").get<float>();
            m_dynamic_regulation[sementic_id] = rigidness;
            m_semantic_label[sementic_id] = label;
            if (label > m_max_seg_label)
                m_max_seg_label = label + 1;
            if (rigidness > max_rigidness)
                max_rigidness = rigidness;
            sementic_id++;
        }
        m_semantic_label.resize(sementic_id);
        m_dynamic_regulation[0] = max_rigidness;  // The background is the most rigid
    }

    // Resample-related
    check_and_load(m_use_resample, "use_resample", false);
    if (m_use_resample)
    {
        m_resample_prob = config_json.at("resample_prob");
    }
    else
    {
        m_resample_prob = 1.f;
    }

    // Keypoint-related
    check_and_load(m_use_keypoint, "use_keypoint", false);
    if (m_use_keypoint)
    {
        std::string keypoint_type_string = config_json.at("keypoint_type");
        if (keypoint_type_string == "SuperPoints" || keypoint_type_string == "SuperPoint" || keypoint_type_string == "superpoints" || keypoint_type_string == "superpoints")
        {
            m_keypoint_type = KeyPointType::SuperPoints;
        }
        else if (keypoint_type_string == "R2D2" || keypoint_type_string == "r2d2")
        {
            m_keypoint_type = KeyPointType::R2D2;
        }
        else if (keypoint_type_string == "ORB" || keypoint_type_string == "orb" || keypoint_type_string == "Orb")
        {
            m_keypoint_type = KeyPointType::ORB;
        }
        else
        {
            std::cout << "Unknown keypoint type: " << keypoint_type_string << std::endl;
            exit(-1);
        }
        m_kp_match_ratio_thresh = config_json.at("kp_match_ratio_thresh");
        m_kp_match_dist_thresh = config_json.at("kp_match_dist_thresh");
    }

    // Sys mode selection
    m_measure_mode = config_json.at("measure_mode");
    m_segmentation_mode = config_json.at("segmentation_mode");
    m_opticalflow_mode = config_json.at("opticalflow_mode");
    m_nodeflow_mode = config_json.at("nodeflow_mode");
    m_optimization_mode = config_json.at("optimization_mode");
    m_dynamic_geometry_mode = config_json.at("dynamic_geometry_mode");
    m_monitor_mode = config_json.at("monitor_mode");

    // Network related
    m_nn_device = config_json.at("nn_device");
    check_and_load_string(m_segmentation_model_specify, "segmentation_model_specify", "segmenter_320x240_model");

    // Process related
    m_reinit_counter = config_json.at("reinit_counter");

    // Eval related
    m_track_semantic_label = config_json.at("track_semantic_label");

    // Log out information
    Log();
}

void star::ConfigParser::loadContextConfigFromJson(const void *json_ptr)
{
    const auto &config_json = *((const json *)json_ptr);
    if (config_json.contains("cam-00"))
    { // Check if camera exists
        loadCameraConfigFromJson(json_ptr);
    }
    else
    {
        printf("Camera info is not detected.\n");
    }
}

void star::ConfigParser::loadCameraConfigFromJson(const void *json_ptr)
{
    const auto &config_json = *((const json *)json_ptr);

    for (auto cam_idx = 0; cam_idx < d_max_cam; ++cam_idx)
    {
        std::string cam_name = "cam-" + stringFormat("%02d", cam_idx);
        if (!config_json.contains(cam_name))
        {
            m_num_cam = cam_idx;
            break;
        }
        const auto &cam_json = config_json.at(cam_name);
        // Depth image setting
        m_raw_img_rows[cam_idx] = cam_json.at("image_rows");
        m_raw_img_cols[cam_idx] = cam_json.at("image_cols");
        m_downsample_scale[cam_idx] = cam_json.at("downsample_scale");
        m_downsample_image_rows[cam_idx] = std::floor(m_downsample_scale[cam_idx] * float(m_raw_img_rows[cam_idx]));
        m_downsample_image_cols[cam_idx] = std::floor(m_downsample_scale[cam_idx] * float(m_raw_img_cols[cam_idx]));
        m_clip_near[cam_idx] = cam_json.at("clip_near");
        m_clip_far[cam_idx] = cam_json.at("clip_far");
        m_depth_flip[cam_idx] = cam_json.at("depth_flip");

        // Depth & Color currently share the same intrinsic (Because we can align them)
        std::vector<float> intrinsic = cam_json.at("intrinsic").get<std::vector<float>>();
        raw_depth_intrinsic[cam_idx].focal_x = intrinsic[0];
        raw_depth_intrinsic[cam_idx].focal_y = intrinsic[1];
        raw_depth_intrinsic[cam_idx].principal_x = intrinsic[2];
        raw_depth_intrinsic[cam_idx].principal_y = intrinsic[3];

        raw_rgb_intrinsic[cam_idx].focal_x = intrinsic[0];
        raw_rgb_intrinsic[cam_idx].focal_y = intrinsic[1];
        raw_rgb_intrinsic[cam_idx].principal_x = intrinsic[2];
        raw_rgb_intrinsic[cam_idx].principal_y = intrinsic[3];

        // The downsample intrinsic
        const float principal_x_downsample = raw_rgb_intrinsic[cam_idx].principal_x * m_downsample_scale[cam_idx];
        const float principal_y_downsample = raw_rgb_intrinsic[cam_idx].principal_y * m_downsample_scale[cam_idx];
        const float focal_x_downsample = raw_rgb_intrinsic[cam_idx].focal_x * m_downsample_scale[cam_idx];
        const float focal_y_downsample = raw_rgb_intrinsic[cam_idx].focal_y * m_downsample_scale[cam_idx];
        downsample_rgb_intrinsic[cam_idx] = Intrinsic(
            focal_x_downsample, focal_y_downsample, principal_x_downsample, principal_y_downsample);

        // Extrinsic
        const json matrix_json = cam_json.at("extrinsic");
        Eigen::Matrix4f cam2world;
        for (auto i = 0; i < 4; ++i)
        {
            for (auto j = 0; j < 4; ++j)
            {
                cam2world(i, j) = matrix_json[size_t(4 * j + i)].get<float>(); // ColMajor
            }
        }
        // Check z-axis-flip
        if (m_depth_flip[cam_idx])
        {
            std::string flip_mod = "blender";
            if (flip_mod == "blender")
            { // Blender is y-z flip
                Eigen::Matrix4f z_flip_matrix;
                z_flip_matrix << 1.f, 0.f, 0.f, 0.f,
                    0.f, -1.f, 0.f, 0.f,
                    0.f, 0.f, -1.f, 0.f,
                    0.f, 0.f, 0.f, 1.f;
                cam2world = cam2world * z_flip_matrix;
            }
        }
        m_cam2world[cam_idx] = cam2world;
    }
}

void star::ConfigParser::Log() const
{
    std::cout << "Config loaded!" << std::endl;
    std::cout << "Num frame: " << m_num_frames << std::endl;
}

void star::ConfigParser::loadVisConfigFromJson(const void *json_ptr)
{
    const auto &config_json = *((const json *)json_ptr);

    m_enable_vis = config_json.at("enable_vis");
    m_pcd_size = config_json.at("pcd_size");
    m_graph_node_size = config_json.at("graph_node_size");
}