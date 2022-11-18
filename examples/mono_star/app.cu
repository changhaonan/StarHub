#include "common/ConfigParser.h"
#include "measure/MeasureProcessorOffline.h"
#include "measure/OpticalFlowProcessorGMA.h"
#include "measure/NodeMotionProcessor.h"
#include "geometry/DynamicGeometryProcessor.h"
#include "opt/OptimizationProcessorWarpSolver.h"
#include <star/math/DualQuaternion.hpp>
#include <star/visualization/Visualizer.h>
// Viewer
#include <easy3d_viewer/context.hpp>

int main()
{
    // Checking
    std::cout << "OpenCV version : " << CV_VERSION << std::endl;
    std::cout << "Major version : " << CV_MAJOR_VERSION << std::endl;
    std::cout << "Minor version : " << CV_MINOR_VERSION << std::endl;
    std::cout << "Subminor version : " << CV_SUBMINOR_VERSION << std::endl;

    using namespace star;
    std::string scene_name = "move_dragon";

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
    auto geometry_processor = std::make_shared<DynamicGeometryProcessor>();
    auto opticalflow_processor = std::make_shared<OpticalFlowProcessorGMA>();
    auto node_motion_processor = std::make_shared<NodeMotionProcessor>();
    auto opt_processor = std::make_shared<OptimizationProcessorWarpSolver>();

    for (int frame_idx = 0; frame_idx < config.num_frames(); frame_idx++)
    {
        // Prepare context
        auto &context = easy3d::Context::Instance();
        context.open(frame_idx);

        // Measure process
        measure_processor->ProcessFrame(frame_idx, 0);

        if (frame_idx > 0)
        {
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
                geometry_processor->ActiveNodeGraph()->GetReferenceNodeCoordinate(),
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

            // Solve
            opt_processor->ProcessFrame(
                measure4solver,
                render4solver,
                geometry4solver,
                node_graph4solver,
                nodeflow4solver,
                opticalflow4solver,
                frame_idx,
                0);

            // Apply the warp
            geometry_processor->ProcessFrame(
                opt_processor->SolvedSE3(), frame_idx, 0); // Dynamic geometry process
        }

        if (frame_idx == 0)
        {
            geometry_processor->initGeometry(*measure_processor->SurfelMapReadOnly(), config.extrinsic()[0], frame_idx, 0);
            GArrayView<DualQuaternion> empty_se3;
            geometry_processor->ProcessFrame(empty_se3, frame_idx, 0);
        }
        // Clean
        context.close();
    }
}