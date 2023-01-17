#include "KeyPointMatcher.h"
#include <cmath>
#include <star/math/vector_ops.hpp>

void star::MatchKeyPointsBFOpenCV(
    const KeyPoints &keypoints_query,
    const KeyPoints &keypoints_train,
    GArraySlice<int2> matches,
    unsigned &num_valid_match,
    const float ratio_thresh,
    const float dist_thresh,
    cudaStream_t stream)
{
    // Download descriptorss
    std::vector<unsigned char> descriptor_query_vec;
    std::vector<unsigned char> descriptor_train_vec;
    keypoints_query.DescriptorReadOnly().Download(descriptor_query_vec);
    keypoints_train.DescriptorReadOnly().Download(descriptor_train_vec);

    // Download vertex
    std::vector<float4> vertex_confid_query_vec;
    std::vector<float4> vertex_confid_train_vec;
    keypoints_query.ReferenceVertexConfidenceReadOnly().Download(vertex_confid_query_vec);
    keypoints_train.ReferenceVertexConfidenceReadOnly().Download(vertex_confid_train_vec);

    // Create cv::Mat from std::vector<float>
    cv::Mat descriptor_query_mat(keypoints_query.NumKeyPoints(), keypoints_query.DescriptorDim(), CV_8U, descriptor_query_vec.data());
    cv::Mat descriptor_train_mat(keypoints_train.NumKeyPoints(), keypoints_train.DescriptorDim(), CV_8U, descriptor_train_vec.data());

    // Match
    cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptor_query_mat, descriptor_train_mat, knn_matches, 2);

    // Filter matches using the Lowe's ratio test
    std::vector<int2> good_matches_int2;
    for (auto j = 0; j < knn_matches.size(); j++)
    {
        if (knn_matches[j].size() >= 2)
        {
            if (knn_matches[j][0].distance < ratio_thresh * knn_matches[j][1].distance &&
                vertex_confid_query_vec[knn_matches[j][0].queryIdx].w > 0.0f &&
                vertex_confid_train_vec[knn_matches[j][0].trainIdx].w > 0.0f &&
                norm(vertex_confid_query_vec[knn_matches[j][0].queryIdx] - vertex_confid_train_vec[knn_matches[j][0].trainIdx]) < dist_thresh)
            {
                good_matches_int2.push_back(make_int2(knn_matches[j][0].queryIdx, knn_matches[j][0].trainIdx));
            }
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

void star::MatchKeyPointsBFOpenCVHostOnly(
    const cv::Mat &keypoints_query,
    const cv::Mat &keypoints_train,
    const cv::Mat &descriptors_query,
    const cv::Mat &descriptors_train,
    GArraySlice<int2> matches,
    unsigned &num_valid_match,
    const float ratio_thresh,
    const float dist_thresh,
    cudaStream_t stream)
{
    // Match
    cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors_query, descriptors_train, knn_matches, 2);

    // Filter matches using the Lowe's ratio test
    std::vector<int2> good_matches_int2;
    for (auto j = 0; j < knn_matches.size(); j++)
    {
        if (knn_matches[j].size() >= 2)
        {
            float2 kp_query = keypoints_query.at<float2>(knn_matches[j][0].queryIdx, 0);
            float2 kp_train = keypoints_train.at<float2>(knn_matches[j][0].trainIdx, 0);
            float kp_dist = norm(kp_query - kp_train);
            if (knn_matches[j][0].distance < ratio_thresh * knn_matches[j][1].distance && kp_dist < dist_thresh)
            {
                good_matches_int2.push_back(make_int2(knn_matches[j][0].queryIdx, knn_matches[j][0].trainIdx));
            }
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

void star::MatchKeyPointsBFOpenCVHostOnly(
    const cv::Mat &keypoints_query,
    const cv::Mat &keypoints_train,
    const cv::Mat &descriptors_query,
    const cv::Mat &descriptors_train,
    GArrayView<float4> vertex_confid_query,
    GArrayView<float4> vertex_confid_train,
    GArraySlice<int2> matches,
    unsigned &num_valid_match,
    const float ratio_thresh,
    const float pixel_dist_thresh,
    const float vertex_dist_thresh,
    cudaStream_t stream)
{
    // Download data
    std::vector<float4> h_vertex_query;
    std::vector<float4> h_vertex_train;
    vertex_confid_query.Download(h_vertex_query);
    vertex_confid_train.Download(h_vertex_train);

    // Match
    cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors_query, descriptors_train, knn_matches, 2);

    // Filter matches using the Lowe's ratio test
    std::vector<int2> good_matches_int2;
    for (auto j = 0; j < knn_matches.size(); j++)
    {
        if (knn_matches[j].size() >= 2)
        {
            // Pixel dist
            float2 kp_pixel_query = keypoints_query.at<float2>(knn_matches[j][0].queryIdx, 0);
            float2 kp_pixel_train = keypoints_train.at<float2>(knn_matches[j][0].trainIdx, 0);
            float kp_pixel_dist = norm(kp_pixel_query - kp_pixel_train);
            // Vertex dist
            float4 kp_vertex_query = h_vertex_query[knn_matches[j][0].queryIdx];
            float4 kp_vertex_train = h_vertex_train[knn_matches[j][0].trainIdx];
            float kp_vertex_dist = std::sqrt(squared_norm_xyz(kp_vertex_query - kp_vertex_train));
            if (knn_matches[j][0].distance < ratio_thresh * knn_matches[j][1].distance && kp_pixel_dist < pixel_dist_thresh && kp_vertex_dist < vertex_dist_thresh)
            {
                good_matches_int2.push_back(make_int2(knn_matches[j][0].queryIdx, knn_matches[j][0].trainIdx));
            }
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