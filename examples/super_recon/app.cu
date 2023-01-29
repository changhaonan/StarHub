#include "common/ConfigParser.h"
// Viewer
#include <easy3d_viewer/context.hpp>
// Config
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
    using namespace star;
    namespace po = boost::program_options;
    
    po::options_description desc("Allowed options");
    desc.add_options()("help", "produce help message")("scene_name", po::value<std::string>(), "name of scene");
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    std::string scene_name = vm["scene_name"].as<std::string>();
    std::cout << "Start testing Super-Reconstruct on " << scene_name << "..." << std::endl;

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

    return 0;
}