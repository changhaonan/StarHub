#include "common/ConfigParser.h"
#include "measure/MeasureProcessorOffline.h"
#include "measure/OpticalFlowProcessorGMA.h"
#include "measure/OpticalFlowProcessorOffline.h"
#include "measure/NodeMotionProcessor.h"
#include "measure/KeyPointDetectProcessor.h"
#include "measure/SegmentationProcessorOffline.h"
#include "geometry/DynamicGeometryProcessor.h"
#include "opt/OptimizationProcessorWarpSolver.h"
#include <star/math/DualQuaternion.hpp>
#include <star/io/YCBPoseReader.h>
#include <star/geometry/node_graph/Skinner.h>
#include <star/visualization/Visualizer.h>
// Viewer
#include <easy3d_viewer/context.hpp>

int main()
{
    std::cout << "Start testing Mono-STAR" << std::endl;

    using namespace star;
    // std::string scene_name = "move_dragon";
    // std::string scene_name = "home1";
    std::string scene_name = "fastycb1";
    // std::string scene_name = "fastycb2";
    auto root_path_prefix = boost::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
    auto config_path_prefix = root_path_prefix / "data";
    auto output_path = root_path_prefix / "external/Easy3DViewer/public/test_data" / scene_name;

    // Parse it
    auto sys_config_path = config_path_prefix / scene_name / "system.json";
    auto context_config_path = config_path_prefix / scene_name / "context.json";
    auto vis_config_path = config_path_prefix / "visualize.json";
    auto &config = ConfigParser::Instance();
    config.ParseConfig(sys_config_path.string(), context_config_path.string(), vis_config_path.string(), output_path.string());

    // Prepare context
    auto &context = easy3d::Context::Instance();
    context.setDir(output_path.string(), "frame");

    // Build the Measure system (Use serial not parallel)
    // Generate the geometry, node graph, and render in continous time
    auto measure_processor = std::make_shared<MeasureProcessorOffline>();
    auto semantic_processor = std::make_shared<SegmentationProcessorOffline>();
    auto keypoint_processor = std::make_shared<KeyPointDetectProcessor>();

    auto geometry_processor = std::make_shared<DynamicGeometryProcessor>();
    // auto opticalflow_processor = std::make_shared<OpticalFlowProcessorOffline>();
    auto opticalflow_processor = std::make_shared<OpticalFlowProcessorGMA>();
    auto node_motion_processor = std::make_shared<NodeMotionProcessor>();
    auto opt_processor = std::make_shared<OptimizationProcessorWarpSolver>();

    //  Evaluation-related
    auto ycb_pose_reader = std::make_shared<YCBPoseReader>();
    auto pose_gt_path = config_path_prefix / scene_name / "pose_gt.txt";
    if (boost::filesystem::exists(pose_gt_path))
    {
        ycb_pose_reader->Parse(pose_gt_path.string());
    }
    Eigen::Matrix4f gt_pose_init = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f est_pose_init = Eigen::Matrix4f::Identity();
    bool pose_inited = false;

    for (int frame_idx = 0; frame_idx < config.num_frames(); frame_idx++)
    {
        std::cout << "==================== Frame-" << frame_idx << " ====================" << std::endl;
        // Prepare context
        auto &context = easy3d::Context::Instance();
        context.open(frame_idx);

        // Measure process
        measure_processor->ProcessFrame(frame_idx, 0);

        // Semantic process (Expand Measurement)
        semantic_processor->ProcessFrame(measure_processor->GetSurfelMap(), frame_idx, 0);

        if (frame_idx > 0)
        {
            // KeyPoint process (Expand Measurement)
            keypoint_processor->ProcessFrame(
                measure_processor->GetSurfelMapTex(),
                geometry_processor->GetSurfelMapTex(),
                frame_idx, 0);

            // Bind keypoint to node graph
            auto geometry4skinner = keypoint_processor->ModelKeyPoints()->GenerateGeometry4Skinner();
            auto nodegraph4skinner = geometry_processor->ActiveNodeGraph()->GenerateNodeGraph4Skinner();
            Skinner::PerformSkinningFromLive(geometry4skinner, nodegraph4skinner, 0);
            if (config.enable_semantic_surfel())
            {
                Skinner::UpdateSkinnningConnection(geometry4skinner, nodegraph4skinner, 0);
            }

            // Optical flow process
            auto surfel_map_this = measure_processor->GetSurfelMapTex();
            auto surfel_map_prev = geometry_processor->GetSurfelMapTex();
            opticalflow_processor->ProcessFrame(surfel_map_this, surfel_map_prev, frame_idx, 0);

            // Save the surfelmotion
            std::string surfelmotion_name = "surfel_motion";
            context.addPointCloud(surfelmotion_name, surfelmotion_name, config.extrinsic()[0].inverse(), config.pcd_size() * 0.1);
            visualize::SavePointCloudWithNormal(
                geometry_processor->ActiveGeometry()->ReferenceVertexConfidenceReadOnly(),
                opticalflow_processor->GetSurfelMotion(),
                context.at(surfelmotion_name));

            // Node flow process
            node_motion_processor->ProcessFrame(
                surfel_map_this, surfel_map_prev, opticalflow_processor->GetOpticalFlow(),
                geometry_processor->ActiveGeometry()->GenerateGeometry4Solver(),
                geometry_processor->ActiveNodeGraph()->GetNodeSize(),
                frame_idx,
                0);

            // Save the nodemotion
            std::string nodemotion_name = "node_motion";
            context.addPointCloud(nodemotion_name, nodemotion_name, config.extrinsic()[0].inverse(), config.pcd_size());
            visualize::SavePointCloudWithNormal(
                geometry_processor->ActiveNodeGraph()->ReferenceNodeCoordinateReadOnly(),
                node_motion_processor->GetNodeMotionPred(),
                context.at(nodemotion_name));

            // Start the optimization process
            auto geometry4solver = geometry_processor->ActiveGeometry()->GenerateGeometry4Solver();
            auto node_graph4solver = geometry_processor->ActiveNodeGraph()->GenerateNodeGraph4Solver();

            Measure4Solver measure4solver;
            measure4solver.num_cam = 1;
            measure4solver.vertex_confid_map[0] = surfel_map_this.vertex_confid;
            measure4solver.normal_radius_map[0] = surfel_map_this.normal_radius;
            measure4solver.index_map[0] = surfel_map_this.index;

            Render4Solver render4solver;
            render4solver.num_cam = 1;
            render4solver.reference_vertex_map[0] = surfel_map_prev.vertex_confid;
            render4solver.reference_normal_map[0] = surfel_map_prev.normal_radius;
            render4solver.index_map[0] = surfel_map_prev.index;

            OpticalFlow4Solver opticalflow4solver;
            opticalflow4solver.num_cam = 1;
            opticalflow4solver.opticalflow_map[0] = opticalflow_processor->GetOpticalFlow();

            NodeFlow4Solver nodeflow4solver;
            nodeflow4solver.num_node = node_motion_processor->GetNodeMotionPred().Size();
            nodeflow4solver.node_motion_pred = node_motion_processor->GetNodeMotionPred();

            KeyPoint4Solver keypoint4solver;
            keypoint4solver.kp_match = keypoint_processor->GetMatchedKeyPointsReadOnly();
            keypoint4solver.kp_measure_vertex_confid = keypoint_processor->MeasureKeyPoints()->ReferenceVertexConfidenceReadOnly();
            keypoint4solver.kp_measure_normal_radius = keypoint_processor->MeasureKeyPoints()->ReferenceNormalRadiusReadOnly();
            keypoint4solver.kp_vertex_confid = keypoint_processor->ModelKeyPoints()->ReferenceVertexConfidenceReadOnly();
            keypoint4solver.kp_normal_radius = keypoint_processor->ModelKeyPoints()->ReferenceNormalRadiusReadOnly();
            keypoint4solver.kp_knn = keypoint_processor->ModelKeyPoints()->SurfelKNNReadOnly();
            keypoint4solver.kp_knn_spatial_weight = keypoint_processor->ModelKeyPoints()->SurfelKNNSpatialWeightReadOnly();
            keypoint4solver.kp_knn_connect_weight = keypoint_processor->ModelKeyPoints()->SurfelKNNConnectWeightReadOnly();

            // Solve
            opt_processor->ProcessFrame(
                measure4solver,
                render4solver,
                geometry4solver,
                node_graph4solver,
                nodeflow4solver,
                opticalflow4solver,
                keypoint4solver,
                frame_idx,
                0);

            // Apply the warp
            geometry_processor->ProcessFrame(
                measure_processor->GetSurfelMapTex(),
                opt_processor->SolvedSE3(),
                frame_idx,
                0); // Dynamic geometry process
        }
        else if (frame_idx == 0)
        {
            GArrayView<DualQuaternion> empty_se3;
            geometry_processor->ProcessFrame(
                measure_processor->GetSurfelMapTex(),
                empty_se3,
                frame_idx,
                0);
        }
        // Add gt pose if exists
        if (ycb_pose_reader->GetNumPoses())
        {
            if (context.has("average_dq"))
            {
                auto gt_pose = ycb_pose_reader->GetPoses()[frame_idx * config.step_frame() + config.start_frame_idx()];
                if (!pose_inited)
                {
                    auto pose_json = context.of("average_dq");
                    auto est_pose_vec = pose_json["vis"]["coordinate"].get<std::vector<float>>();
                    Eigen::Matrix4f est_pose_mat(est_pose_vec.data());
                    est_pose_init = est_pose_mat;
                    gt_pose_init = gt_pose;
                    pose_inited = true;
                }
                auto transferred_gt_pose = config.extrinsic()[0].inverse() * (gt_pose * gt_pose_init.inverse()) * est_pose_init;
                context.addCoord("gt_pose", "", transferred_gt_pose);
            }
        }
        // Add extra info
        json extra_info;
        extra_info["real_frame_idx"] = frame_idx + config.start_frame_idx();
        context.addExtra("extra_info", extra_info);
        // Clean
        context.close();
    }
}