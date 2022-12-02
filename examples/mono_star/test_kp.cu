#include "common/ConfigParser.h"
#include <star/io/VolumeDeformFileFetch.h>
// Viewer
#include <easy3d_viewer/context.hpp>

int main()
{
    std::cout << "Start testing KeyPoints" << std::endl;
    using namespace star;
    std::string scene_name = "fastycb1";
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

    // Prepare fetcher
    auto fetcher = std::make_shared<VolumeDeformFileFetch>(config.data_path());
    for (auto i = 0; i < config.num_frames(); ++i) {
        auto img_idx = config.start_frame_idx() + config.step_frame() * i;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        auto frame = fetcher->FetchKeypoint(0, img_idx, keypoints, descriptors);
    }

    context.close();
}