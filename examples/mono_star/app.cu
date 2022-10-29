#include "common/ConfigParser.h"
#include "measure/MeasureProcessorOffline.h"
#include "geometry/DynamicGeometryProcessor.h"
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
    // auto dynamic_geometry_processor = std::make_shared<DynamicGeometryProcessor>();

    for (int frame_idx = 0; frame_idx < config.num_frames(); frame_idx++)
    {
        // Prepare context
        auto &context = easy3d::Context::Instance();
        context.open(frame_idx);

        // Measure process
        measure_processor->processFrame(frame_idx, 0);

        // Dynamic geometry process
        // dynamic_geometry_processor->initGeometry(
        //     *measure_processor->SurfelMapReadOnly(),
        //     config.extrinsic()[0],
        //     0);

        // Clean
        context.close();
    }
}