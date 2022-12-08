#include "KeyPointMatcher.h"
#include <star/math/vector_ops.hpp>

void star::MatchKeyPointsBFOpenCV(
    const KeyPoints &key_points_src,
    const KeyPoints &key_points_dst,
    GArraySlice<int2> matches,
    unsigned &num_valid_match,
    const float ratio_thresh,
    const float dist_thresh,
    cudaStream_t stream)
{
    // Download descriptorss
    std::vector<float> descriptor_src_vec;
    std::vector<float> descriptor_dst_vec;
    key_points_src.DescriptorReadOnly().Download(descriptor_src_vec);
    key_points_dst.DescriptorReadOnly().Download(descriptor_dst_vec);

    // Download vertex
    std::vector<float4> vertex_confid_src_vec;
    std::vector<float4> vertex_confid_dst_vec;
    key_points_src.ReferenceVertexConfidenceReadOnly().Download(vertex_confid_src_vec);
    key_points_dst.ReferenceVertexConfidenceReadOnly().Download(vertex_confid_dst_vec);

    // Create cv::Mat from std::vector<float>
    cv::Mat descriptor_src_mat(key_points_src.NumKeyPoints(), key_points_src.DescriptorDim(), CV_32F, descriptor_src_vec.data());
    cv::Mat descriptor_dst_mat(key_points_dst.NumKeyPoints(), key_points_dst.DescriptorDim(), CV_32F, descriptor_dst_vec.data());

    // Match
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptor_src_mat, descriptor_dst_mat, knn_matches, 2);

    // Filter matches using the Lowe's ratio test
    std::vector<int2> good_matches_int2;
    for (auto j = 0; j < knn_matches.size(); j++)
    {
        if (knn_matches[j][0].distance < ratio_thresh * knn_matches[j][1].distance &&
            vertex_confid_src_vec[knn_matches[j][0].queryIdx].w > 0.0f &&
            vertex_confid_dst_vec[knn_matches[j][0].trainIdx].w > 0.0f &&
            norm(vertex_confid_src_vec[knn_matches[j][0].queryIdx] - vertex_confid_dst_vec[knn_matches[j][0].trainIdx]) < dist_thresh)
        {
            good_matches_int2.push_back(make_int2(knn_matches[j][0].queryIdx, knn_matches[j][0].trainIdx));
        }
    }

    // Copy to GPU & Sync
    num_valid_match = good_matches_int2.size();
    cudaSafeCall(cudaMemcpyAsync(
        matches.Ptr(),
        good_matches_int2.data(),
        sizeof(int2) * good_matches_int2.size(),
        cudaMemcpyHostToDevice,
        stream));
    cudaSafeCall(cudaStreamSynchronize(stream));
}