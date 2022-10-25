#pragma once
#include <Eigen/Eigen>
#include <string>
#include <star/common/common_types.h>
#include <star/common/string_utils.hpp>
#include <boost/filesystem.hpp>
#include "global_configs.h"

namespace star
{

    class ConfigParser
    {
    private:
        // Do not allow user-contruct
        explicit ConfigParser();

    public:
        static ConfigParser &Instance();
        void ParseConfig(
            const std::string &sys_config_path,
            const std::string &context_config_path,
            const std::string &vis_config_path,
            const std::string &output_path);
        void Log() const;
        // Path-related
    public:
        std::string output_path() const { return m_output_path; }
        std::string model_path() const { return m_model_path; }

    private:
        std::string m_output_path;
        std::string m_model_path;
        // Context-related
    public:
        void loadContextConfigFromJson(const void *json_ptr);
        void loadCameraConfigFromJson(const void *json_ptr);

    public:
        unsigned num_cam() const { return m_num_cam; }
        const Eigen::Matrix4f *extrinsic() const { return m_cam2world; }
        Intrinsic depth_intrinsic_raw(size_t cam_idx) const { return raw_depth_intrinsic[cam_idx]; }
        Intrinsic rgb_intrinsic_raw(size_t cam_idx) const { return raw_rgb_intrinsic[cam_idx]; }
        Intrinsic rgb_intrinsic_downsample(size_t cam_idx) const { return downsample_rgb_intrinsic[cam_idx]; }

        unsigned raw_img_rows(const size_t cam_idx = 0) const { return m_raw_img_rows[cam_idx]; }
        unsigned raw_img_cols(const size_t cam_idx = 0) const { return m_raw_img_cols[cam_idx]; }
        unsigned downsample_img_rows(const size_t cam_idx = 0) const { return m_downsample_image_rows[cam_idx]; }
        unsigned downsample_img_cols(const size_t cam_idx = 0) const { return m_downsample_image_cols[cam_idx]; }
        float clip_near(const size_t cam_idx = 0) const { return m_clip_near[cam_idx]; }
        float clip_far(const size_t cam_idx = 0) const { return m_clip_far[cam_idx]; }
        float max_rendering_depth(const size_t cam_idx = 0) const { return m_clip_far[cam_idx]; }
        bool depth_flip(const size_t cam_idx = 0) const { return m_depth_flip[cam_idx]; }
        float downsample_scale(const size_t cam_idx = 0) const { return m_downsample_scale[cam_idx]; }

    private:
        unsigned m_num_cam; // Number of valid camera is computed on flight
        Eigen::Matrix4f m_cam2world[d_max_cam];
        Intrinsic raw_depth_intrinsic[d_max_cam];
        Intrinsic raw_rgb_intrinsic[d_max_cam];
        Intrinsic downsample_rgb_intrinsic[d_max_cam];

        float m_clip_near[d_max_cam] = {0.f};
        float m_clip_far[d_max_cam] = {0.f};
        unsigned m_raw_img_rows[d_max_cam] = {0};
        unsigned m_raw_img_cols[d_max_cam] = {0};
        unsigned m_downsample_image_rows[d_max_cam] = {0};
        unsigned m_downsample_image_cols[d_max_cam] = {0};
        bool m_depth_flip[d_max_cam] = {false};
        float m_downsample_scale[d_max_cam] = {0.f}; // Shared across cameras

        // Sys-related
    public:
        void loadSysConfigFromJson(const void *json_ptr);

    private:
        std::string m_data_prefix = "";

    public:
        const boost::filesystem::path data_path() const { return boost::filesystem::path(m_data_prefix); }
        // Frame-related
    private:
        int m_start_frame_idx = 0;
        int m_num_frames = 0;
        int m_step_frame = 0;

    public:
        int start_frame_idx() const { return m_start_frame_idx; }
        int num_frames() const { return m_num_frames; }
        int step_frame() const { return m_step_frame; }
        // Debug-related
    private:
        bool m_debug_node_external_load = false;

    public:
        bool debug_node_external_load() { return m_debug_node_external_load; }

        // Filter-related
    public:
        bool use_gaussian_filter() const { return m_use_gaussian_filter; }

    private:
        bool m_use_gaussian_filter = false;

        // Optical-flow-related
    public:
        float opticalflow_suppress_threshold() const { return m_opticalflow_suppress_threshold; }

    private:
        float m_opticalflow_suppress_threshold = 0.f;

        // Tsdf-related
    public:
        unsigned tsdf_width() const { return m_tsdf_width; }
        unsigned tsdf_height() const { return m_tsdf_height; }
        unsigned tsdf_depth() const { return m_tsdf_depth; }
        float tsdf_voxel_size() const { return m_tsdf_voxel_size_m; }
        float3 tsdf_origin() const { return m_tsdf_origin_m; }

    private:
        unsigned m_tsdf_width = 0;
        unsigned m_tsdf_height = 0;
        unsigned m_tsdf_depth = 0;
        float m_tsdf_voxel_size_m = 0.f;                     // (meter)
        float3 m_tsdf_origin_m = make_float3(0.f, 0.f, 0.f); // (meter)

        // Surfel&NodeGraph-related
    public:
        float node_radius() const { return m_node_radius; }
        float surfel_radius_scale() const { return m_surfel_radius_scale; }
        bool enable_semantic_surfel() const { return m_enable_semantic_surfel; }

    private:
        float m_node_radius = 0.f;
        float m_surfel_radius_scale = 0.f;
        bool m_enable_semantic_surfel = false;

        // Object-specified
    public:
        floatX<d_max_num_semantic> dynamic_regulation() const { return m_dynamic_regulation; }

    private:
        // Dynamic regulation, apply a different cofficient to different type of objects
        floatX<d_max_num_semantic> m_dynamic_regulation;

        // Sample-related
    public:
        bool use_resample() const { return m_use_resample; }
        float resample_prob() const { return m_resample_prob; }

    private:
        bool m_use_resample = false;
        float m_resample_prob = false;

        // Processor selection
    public:
        std::string measure_mode() const { return m_measure_mode; }
        std::string segmentation_mode() const { return m_segmentation_mode; }
        std::string opticalflow_mode() const { return m_opticalflow_mode; }
        std::string nodeflow_mode() const { return m_nodeflow_mode; }
        std::string optimization_mode() const { return m_optimization_mode; }
        std::string dynamic_geometry_mode() const { return m_dynamic_geometry_mode; }
        std::string monitor_mode() const { return m_monitor_mode; }

    private:
        std::string m_measure_mode;
        std::string m_segmentation_mode;
        std::string m_opticalflow_mode;
        std::string m_nodeflow_mode;
        std::string m_optimization_mode;
        std::string m_dynamic_geometry_mode;
        std::string m_monitor_mode;

        // Network-related
    public:
        std::string nn_device() { return m_nn_device; }
        std::string segmentation_model_specify() { return m_segmentation_model_specify; }

    private:
        std::string m_nn_device = "";
        std::string m_segmentation_model_specify = "";

        // Visualization related
    public:
        void loadVisConfigFromJson(const void *json_ptr);

    public:
        float pcd_size() const { return m_pcd_size; }
        float graph_node_size() const { return m_graph_node_size; }

    private:
        float m_pcd_size = 0.f;
        float m_graph_node_size = 0.f;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}