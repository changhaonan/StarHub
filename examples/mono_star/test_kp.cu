#include "common/ConfigParser.h"
#include <star/io/VolumeDeformFileFetch.h>
#include <opencv2/features2d.hpp>
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
    cv::Mat last_descriptor;
    cv::Mat last_rgb_img;
    std::vector<cv::KeyPoint> last_keypoints_vec;
    
    auto fetcher = std::make_shared<VolumeDeformFileFetch>(config.data_path());
    for (auto i = 0; i < config.num_frames(); ++i) {
        auto img_idx = config.start_frame_idx() + config.step_frame() * i;
        cv::Mat keypoints;
        cv::Mat descriptors;
        cv::Mat rgb_img;
        fetcher->FetchKeypoint(0, img_idx, keypoints, descriptors, KeyPointType::R2D2);
        fetcher->FetchRGBImage(0, img_idx, rgb_img);

        // Transfer cv::Mat into std::vector<cv::KeyPoint>
        std::vector<cv::KeyPoint> keypoints_vec;
        for (auto i = 0; i < keypoints.rows; ++i) {
            cv::Point2f pt(keypoints.at<float>(i, 0), keypoints.at<float>(i, 1));
            cv::KeyPoint kp;
            kp.pt = pt;
            keypoints_vec.emplace_back(kp);
        }

        if (i > 0) {
            // Matching descriptor vectors with a FLANN based matcher
            // Since SURF is a floating-point descriptor NORM_L2 is used
            cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher->knnMatch(last_descriptor, descriptors, knn_matches, 2 );

            // Filter matches using the Lowe's ratio test
            const float ratio_thresh = 0.7f;  // 0.7f
            std::vector<cv::DMatch> good_matches;
            for (auto j = 0; j < knn_matches.size(); j++)
            {
                if (knn_matches[j][0].distance < ratio_thresh * knn_matches[j][1].distance)
                {
                    good_matches.push_back(knn_matches[j][0]);
                    if (good_matches.size() > 500) break;
                }
            }

            // Draw matches
            cv::Mat img_matches;
            cv::drawMatches(last_rgb_img, last_keypoints_vec, rgb_img, keypoints_vec, good_matches, img_matches, cv::Scalar::all(-1),
                cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            
            cv::imshow("matches", img_matches);
            cv::waitKey(0);
        }
        
        // Copy image
        last_rgb_img = rgb_img;
        last_descriptor = descriptors;
        last_keypoints_vec = keypoints_vec;
    }

    context.close();
}