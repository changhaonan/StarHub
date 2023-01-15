#pragma once
// Read YCB pose file
#include <string>
#include <fstream>
#include <iostream>
#include <Eigen/Eigen>
#include <memory>

namespace star
{
    /**
     * @brief Read YCB pose file
     * @note YCB pose file contains multiple lines, each line contains (Qx, Qy, Qz, Qw, Tx, Ty, Tz)
    */
    class YCBPoseReader
    {
    public:
        using Ptr = std::shared_ptr<YCBPoseReader>;
        YCBPoseReader() : m_num_poses(0) {}
        void Parse(const std::string &file_path);
        Eigen::Matrix4f ParsePose(const std::string &line);
        unsigned GetNumPoses() const { return m_num_poses; }
        const std::vector<Eigen::Matrix4f> &GetPoses() const { return m_poses; }
    private:
        unsigned m_num_poses;
        std::vector<Eigen::Matrix4f> m_poses;
    };
}