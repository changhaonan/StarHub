#include "YCBPoseReader.h"

void star::YCBPoseReader::Parse(const std::string &file_path)
{
    // Reset
    m_poses.clear();
    m_num_poses = 0;
    std::ifstream file_handle(file_path);
    // Read line by line
    std::string line;
    for (std::string line; std::getline(file_handle, line); m_num_poses++)
    {
        m_poses.push_back(ParsePose(line));
    }
}

Eigen::Matrix4f star::YCBPoseReader::ParsePose(const std::string &line)
{
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    std::stringstream ss(line);
    std::string token;
    std::vector<float> tokens;
    while (std::getline(ss, token, ' '))
    {
        tokens.push_back(std::stof(token));
    }
    pose.block<3, 1>(0, 3) = Eigen::Vector3f(tokens[0], tokens[1], tokens[2]);
    pose.block<3, 3>(0, 0) = Eigen::AngleAxisf(tokens[6], Eigen::Vector3f(tokens[3], tokens[4], tokens[5])).toRotationMatrix();
    return pose;
}