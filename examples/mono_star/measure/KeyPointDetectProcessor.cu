#include "KeyPointDetectProcessor.h"
#include <star/geometry/surfel/SurfelGeometryInitializer.h>
#include <star/geometry/keypoint/KeyPointMatcher.h>
#include <star/visualization/Visualizer.h>

namespace star::device
{
    __global__ void GetMatchedPointKernel(
        const float4 *__restrict__ vertex_confid_src,
        const float4 *__restrict__ vertex_confid_dst,
        float4 *__restrict__ matched_vertex_confid_src,
        float4 *__restrict__ matched_vertex_confid_dst,
        int2 *__restrict__ matches,
        const unsigned num_matches)
    {
        const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_matches)
            return;

        const int2 match = matches[idx];
        matched_vertex_confid_src[idx] = vertex_confid_src[match.x];
        matched_vertex_confid_dst[idx] = vertex_confid_dst[match.y];
    }

    // Here rgbd is (-1, 1) transfer them to (0, 255)
    __global__ void RGBDToRGBKernel(
        cudaTextureObject_t rgbd_tex,
        uchar3 *__restrict__ rgb,
        const unsigned img_width,
        const unsigned img_height)
    {
        const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
        const unsigned idy = threadIdx.y + blockIdx.y * blockDim.y;
        if (idx >= img_width || idy >= img_height)
            return;
        float4 rbgd = tex2D<float4>(rgbd_tex, idx, idy);
        uchar3 rgb_uchar3;
        rgb_uchar3.x = (unsigned char)((rbgd.x + 1) * 127.5);
        rgb_uchar3.y = (unsigned char)((rbgd.y + 1) * 127.5);
        rgb_uchar3.z = (unsigned char)((rbgd.z + 1) * 127.5);
        rgb[idy * img_width + idx] = rgb_uchar3;
    }

}

star::KeyPointDetectProcessor::KeyPointDetectProcessor()
    : m_buffer_idx(0), m_num_valid_matches(0)
{
    std::cout << "Initilize KeyPointDetectProcessor..." << std::endl;
    auto &config = ConfigParser::Instance();
    m_keypoint_type = config.keypoint_type();

    m_fetcher = std::make_shared<VolumeDeformFileFetch>(config.data_path());

    // Camera parameters
    m_step_frame = config.step_frame();
    m_start_frame_idx = config.start_frame_idx();
    m_cam2world = config.extrinsic()[0];
    m_enable_semantic_surfel = config.enable_semantic_surfel();
    m_downsample_scale = config.downsample_scale();
    m_image_width = config.downsample_img_cols();
    m_image_height = config.downsample_img_rows();

    // Create host buffer
    cudaSafeCall(cudaMallocHost(&m_keypoint_buffer, sizeof(float) * 2 * d_max_num_keypoints));
    cudaSafeCall(cudaMallocHost(&m_descriptor_buffer, sizeof(float) * d_max_num_keypoints * KeyPoints::GetDescriptorDim(m_keypoint_type)));

    unsigned num_pixels = m_image_width * m_image_height;
    cudaSafeCall(cudaMallocHost(&m_h_rgb, sizeof(uchar3) * num_pixels));
    m_g_rgb.create(num_pixels * sizeof(uchar3));

    m_g_keypoints.AllocateBuffer(d_max_num_keypoints);
    m_keypoint_matches.AllocateBuffer(d_max_num_keypoints);

    m_detected_keypoints = std::make_shared<star::KeyPoints>(config.keypoint_type());

    m_matched_vertex_src.AllocateBuffer(d_max_num_keypoints);
    m_matched_vertex_dst.AllocateBuffer(d_max_num_keypoints);

    // Matching related
    m_kp_match_ratio_thresh = config.kp_match_ratio_thresh();
    m_kp_match_dist_thresh = config.kp_match_dist_thresh();

    // Vis
    m_enable_vis = config.enable_vis();
    m_pcd_size = config.pcd_size();
}

star::KeyPointDetectProcessor::~KeyPointDetectProcessor()
{
    cudaSafeCall(cudaFreeHost(m_keypoint_buffer));
    cudaSafeCall(cudaFreeHost(m_descriptor_buffer));
    cudaSafeCall(cudaFreeHost(m_h_rgb));
    m_g_rgb.release();

    m_g_keypoints.ReleaseBuffer();
    m_keypoint_matches.ReleaseBuffer();

    m_matched_vertex_src.ReleaseBuffer();
    m_matched_vertex_dst.ReleaseBuffer();
}

void star::KeyPointDetectProcessor::ProcessFrame(
    const SurfelMapTex &measure_surfel_map,
    const SurfelMapTex &model_surfel_map,
    const KeyPoints &model_keypoints,
    const unsigned frame_idx,
    cudaStream_t stream)
{
    // Detect Feature
    detectFeature(measure_surfel_map, m_keypoint_tar, m_descriptor_tar, stream);
    if (frame_idx > 0)
    {
        detectFeature(model_surfel_map, m_keypoint_src, m_descriptor_src, stream);
        MatchKeyPointsBFOpenCVHostOnly(
            m_keypoint_src,   // Model
            m_keypoint_tar,   // Measure
            m_descriptor_src, // Model
            m_descriptor_tar, // Measure
            m_keypoint_matches.Slice(),
            m_num_valid_matches,
            m_kp_match_ratio_thresh,
            m_kp_match_dist_thresh,
            stream);
        m_keypoint_matches.ResizeArrayOrException(m_num_valid_matches);
    }

    const auto image_idx = size_t(frame_idx) * m_step_frame + m_start_frame_idx;
    m_fetcher->FetchKeypoint(0, image_idx, m_keypoint_tar, m_descriptor_tar, m_keypoint_type);

    // Scale keypoints according to downsample ratio
    m_keypoint_tar = m_keypoint_tar * m_downsample_scale;

    // Load the keypoint and descriptor into gpu
    unsigned num_keypoints_detected = m_keypoint_tar.rows;
    // CPU copy
    memcpy(m_keypoint_buffer, m_keypoint_tar.data,
           sizeof(float) * m_keypoint_tar.total());
    memcpy(m_descriptor_buffer, m_descriptor_tar.data,
           sizeof(float) * m_descriptor_tar.total());

    // Copy to GPU
    cudaSafeCall(cudaMemcpyAsync(
        m_g_keypoints.Ptr(),
        m_keypoint_buffer,
        m_keypoint_tar.total() * sizeof(float),
        cudaMemcpyHostToDevice,
        stream));
    cudaSafeCall(cudaMemcpyAsync(
        m_detected_keypoints->Descriptor().Ptr(),
        m_descriptor_buffer,
        m_descriptor_tar.total() * sizeof(float),
        cudaMemcpyHostToDevice,
        stream));

    // Resize
    m_g_keypoints.ResizeArrayOrException(num_keypoints_detected);
    m_detected_keypoints->Resize(num_keypoints_detected);

    // Build 3d keypoint
    // Init keypoint geometry
    SurfelGeometryInitializer::InitFromGeometryMap(
        *m_detected_keypoints,
        measure_surfel_map,
        m_g_keypoints.View(),
        m_cam2world,
        m_enable_semantic_surfel,
        stream);

    if (frame_idx != 0 && m_detected_keypoints->NumKeyPoints() > 0 && model_keypoints.NumKeyPoints() > 0)
    {
        // Apply matching between model keypoints and detected keypoints
        MatchKeyPointsBFOpenCV(
            *m_detected_keypoints,
            model_keypoints,
            m_keypoint_matches.Slice(),
            m_num_valid_matches,
            m_kp_match_ratio_thresh,
            m_kp_match_dist_thresh,
            stream);
        m_keypoint_matches.ResizeArrayOrException(m_num_valid_matches);
    }

    // Sync stream
    cudaSafeCall(cudaStreamSynchronize(stream));

    // Save context
    if (m_enable_vis)
        saveContext(model_keypoints, frame_idx, stream);
}

void star::KeyPointDetectProcessor::saveContext(
    const KeyPoints &model_keypoints, unsigned frame_idx, cudaStream_t stream)
{
    auto &context = easy3d::Context::Instance();
    context.addPointCloud("d_keypoints", "", Eigen::Matrix4f::Identity(), m_pcd_size);
    visualize::SavePointCloud(
        m_detected_keypoints->LiveVertexConfidenceReadOnly(),
        context.at("d_keypoints"));

    if (model_keypoints.NumKeyPoints() > 0)
    {
        context.addPointCloud("model_keypoints", "", Eigen::Matrix4f::Identity(), m_pcd_size);
        visualize::SavePointCloud(
            model_keypoints.LiveVertexConfidenceReadOnly(),
            context.at("model_keypoints"));
    }

    // Draw the matched keypoints
    if (m_num_valid_matches > 0)
    {
        getMatchedKeyPoints(*m_detected_keypoints, model_keypoints, stream);
        context.addPointCloud("matched_keypoints", "", Eigen::Matrix4f::Identity(), m_pcd_size);
        visualize::SaveMatchedPointCloud(
            m_matched_vertex_src.View(),
            m_matched_vertex_dst.View(),
            context.at("matched_keypoints"));
    }
}

void star::KeyPointDetectProcessor::getMatchedKeyPoints(
    const KeyPoints &keypoints_src,
    const KeyPoints &keypoints_dst,
    cudaStream_t stream)
{
    // Resize
    m_matched_vertex_src.ResizeArrayOrException(m_num_valid_matches);
    m_matched_vertex_dst.ResizeArrayOrException(m_num_valid_matches);

    // Get matched keypoints
    dim3 blk(128);
    dim3 grid(divUp(m_num_valid_matches, blk.x));
    device::GetMatchedPointKernel<<<grid, blk, 0, stream>>>(
        keypoints_src.ReferenceVertexConfidenceReadOnly().Ptr(),
        keypoints_dst.ReferenceVertexConfidenceReadOnly().Ptr(),
        m_matched_vertex_src.Ptr(),
        m_matched_vertex_dst.Ptr(),
        m_keypoint_matches.Ptr(),
        m_num_valid_matches);
}

void star::KeyPointDetectProcessor::detectFeature(
    const SurfelMapTex &surfel_map, cv::Mat &keypoint, cv::Mat &descriptor, cudaStream_t stream)
{
    if (m_keypoint_type == KeyPointType::ORB)
    {
        detectORBFeature(surfel_map.rgbd, keypoint, descriptor, stream);
    }
    else
    {
        throw std::runtime_error("KeyPoint Type not supported!");
    }
}

void star::KeyPointDetectProcessor::detectORBFeature(
    cudaTextureObject_t rgbd_tex, cv::Mat &keypoint, cv::Mat &descriptor, cudaStream_t stream)
{
    // Download the image from Texture
    dim3 blk(32, 32);
    dim3 grid(divUp(m_image_width, blk.x), divUp(m_image_height, blk.y));
    device::RGBDToRGBKernel<<<grid, blk, 0, stream>>>(
        rgbd_tex, m_g_rgb.ptr(), m_image_width, m_image_height);

    // Copy from device to host
    cudaSafeCall(cudaMemcpyAsync(
        m_h_rgb, m_g_rgb.ptr(),
        sizeof(uchar3) * m_image_width * m_image_height,
        cudaMemcpyDeviceToHost, stream));
    cudaSafeCall(cudaStreamSynchronize(stream));

    cv::Mat image(m_image_height, m_image_width, CV_8UC3, m_h_rgb);
    // Convert to gray
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_RGB2GRAY);
    // Detect ORB features
    std::vector<cv::KeyPoint> keypoints;
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create();
    orb->detectAndCompute(gray_image, cv::Mat(), keypoints, descriptor);
    // Save keypoints into mat
    keypoint = cv::Mat(keypoints.size(), 1, CV_32FC2);
    for (int i = 0; i < keypoints.size(); i++)
    {
        keypoint.at<float2>(i, 0) = make_float2(keypoints[i].pt.x, keypoints[i].pt.y);
    }

    // Log info
    LOG(INFO) << "Number of keypoints detected: " << keypoints.size();
}