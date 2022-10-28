#include "common/ConfigParser.h"
#include "measure/MeasureProcessorOffline.h"
// Viewer
#include <easy3d_viewer/context.hpp>

int main()
{
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
    auto& context = easy3d::Context::Instance();
    context.setDir(output_path.string(), "frame");

    // Build the Measure system (Use serial not parallel)
    // Generate the geometry, node graph, and render in continous time
    auto measure_processor = std::make_shared<MeasureProcessorOffline>();

    for (int i = 0; i < config.num_frames(); i++)
    {
        measure_processor->processFrame(i, 0);
    }
}