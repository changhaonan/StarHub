#include "KeyPointFusor.h"
#include <star/math/vector_ops.hpp>

namespace star::device
{
    __global__ void ResetIndicatorKernel(
        unsigned *__restrict__ not_matched_indicator,
        const int num_new_keypoints)
    {
        const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_new_keypoints)
            return;
        not_matched_indicator[idx] = 1;
    }

    __global__ void MarkMatchAndReplaceKernel(
        const float4 *__restrict__ old_kp_vertrex,
        const float4 *__restrict__ new_kp_vertrex,
        unsigned char *__restrict__ old_kp_descriptor,
        const unsigned char *__restrict__ new_kp_descriptor,
        const int2 *__restrict__ potential_matches,
        unsigned *__restrict__ not_matched_indicator,
        const float kp_match_threshold,
        const int num_matches,
        const int dim_descriptor)
    {
        const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_matches)
            return;
        // Prepare
        const auto match = potential_matches[idx]; // (new, old)
        const auto old_kp = old_kp_vertrex[match.y];
        const auto new_kp = new_kp_vertrex[match.x];
        unsigned char *old_kp_desc_ptr = old_kp_descriptor + match.y * dim_descriptor;
        const unsigned char *new_kp_desc_ptr = new_kp_descriptor + match.x * dim_descriptor;
        const auto dist = norm(old_kp - new_kp);

        // Update indicator
        not_matched_indicator[match.x] = 0;

        // Update descriptor
        if (dist < kp_match_threshold)
        {
            for (auto i = 0; i < dim_descriptor; ++i)
            {
                old_kp_desc_ptr[i] = new_kp_desc_ptr[i];
            }
        }
    }

    __global__ void AppendNewKeyPointsKernel(
        float4 *__restrict__ old_kp_vertrex_confid,
        const float4 *__restrict__ new_kp_vertrex_confid,
        float4 *__restrict__ old_kp_normal_radius,
        const float4 *__restrict__ new_kp_normal_radius,
        float4 *__restrict__ old_kp_color_time,
        const float4 *__restrict__ new_kp_color_time,
        ucharX<d_max_num_semantic> *__restrict__ old_kp_semantic_prob,
        const ucharX<d_max_num_semantic> *__restrict__ new_kp_semantic_prob,
        unsigned char *__restrict__ old_kp_descriptor,
        const unsigned char *__restrict__ new_kp_descriptor,
        const unsigned *__restrict__ not_matched_indicator,
        const unsigned *__restrict__ not_matched_offset,
        const unsigned num_new_kp,
        const unsigned num_old_kp,
        const int dim_descriptor)
    {
        const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_new_kp)
            return;

        if (not_matched_indicator[idx])
        {
            // Append kp info
            const auto offset = not_matched_offset[idx];
            old_kp_vertrex_confid[num_old_kp + offset] = new_kp_vertrex_confid[idx];
            old_kp_normal_radius[num_old_kp + offset] = new_kp_normal_radius[idx];
            old_kp_color_time[num_old_kp + offset] = new_kp_color_time[idx];
            old_kp_semantic_prob[num_old_kp + offset] = new_kp_semantic_prob[idx];
            // Append kp descriptor
            for (auto i = 0; i < dim_descriptor; ++i)
            {
                old_kp_descriptor[(num_old_kp + offset) * dim_descriptor + i] =
                    new_kp_descriptor[idx * dim_descriptor + i];
            }
        }
    }
}

star::KeyPointFusor::KeyPointFusor(
    const float kp_match_threshold) : m_kp_match_threshold(kp_match_threshold)
{
    m_not_matched_indicator.AllocateBuffer(d_max_num_keypoints);
    m_not_matched_prefix_sum.AllocateBuffer(d_max_num_keypoints);
    cudaSafeCall(cudaMallocHost(&m_host_num_not_matches, sizeof(unsigned)));
}

star::KeyPointFusor::~KeyPointFusor()
{
    m_not_matched_indicator.ReleaseBuffer();
    cudaSafeCall(cudaFreeHost(m_host_num_not_matches));
}

void star::KeyPointFusor::Fuse(
    KeyPoints::Ptr old_keypoints,
    KeyPoints::Ptr new_keypoints,
    GArrayView<int2> matches,
    const bool enable_append,
    cudaStream_t stream)
{
    STAR_CHECK_NE(old_keypoints->NumKeyPoints(), 0);
    STAR_CHECK_NE(new_keypoints->NumKeyPoints(), 0);
    // Reset indicator
    resetIdicator(new_keypoints->NumKeyPoints(), stream);

    // Mark match and replace descriptor
    markMatchAndReplace(old_keypoints, new_keypoints, matches, stream);

    // Append the new kp to the back
    if (enable_append)
        appendNewKeyPoints(old_keypoints, new_keypoints, stream);
}

void star::KeyPointFusor::resetIdicator(
    const unsigned num_new_keypoints,
    cudaStream_t stream)
{
    dim3 blk(256);
    dim3 grid(divUp(num_new_keypoints, blk.x));
    device::ResetIndicatorKernel<<<grid, blk, 0, stream>>>(
        m_not_matched_indicator.Ptr(),
        num_new_keypoints);
    m_not_matched_indicator.ResizeArrayOrException(num_new_keypoints);
}

void star::KeyPointFusor::markMatchAndReplace(
    KeyPoints::Ptr old_keypoints,
    KeyPoints::Ptr new_keypoints,
    GArrayView<int2> matches,
    cudaStream_t stream)
{
    const auto num_matches = matches.Size();
    dim3 blk(256);
    dim3 grid(divUp(num_matches, blk.x));
    device::MarkMatchAndReplaceKernel<<<grid, blk, 0, stream>>>(
        old_keypoints->LiveVertexConfidenceReadOnly().Ptr(),
        new_keypoints->LiveVertexConfidenceReadOnly().Ptr(),
        old_keypoints->Descriptor().Ptr(),
        new_keypoints->DescriptorReadOnly().Ptr(),
        matches.Ptr(),
        m_not_matched_indicator.Ptr(),
        m_kp_match_threshold,
        num_matches,
        old_keypoints->DescriptorDim());
    std::cout << "Num_matches: " << num_matches << std::endl;
}

void star::KeyPointFusor::appendNewKeyPoints(
    KeyPoints::Ptr old_keypoints,
    KeyPoints::Ptr new_keypoints,
    cudaStream_t stream)
{
    // Do a prefix sum and compaction
    m_not_matched_prefix_sum.InclusiveSum(m_not_matched_indicator.View(), stream);
    const auto &prefixsum_label = m_not_matched_prefix_sum.valid_prefixsum_array;
    cudaSafeCall(cudaMemcpyAsync(
        (void *)m_host_num_not_matches,
        prefixsum_label.ptr() + prefixsum_label.size() - 1,
        sizeof(unsigned),
        cudaMemcpyDeviceToHost,
        stream));

    const auto num_new_kp = new_keypoints->NumKeyPoints();
    const auto num_old_kp = old_keypoints->NumKeyPoints();
    dim3 blk(256);
    dim3 grid(divUp(num_new_kp, blk.x));
    device::AppendNewKeyPointsKernel<<<grid, blk, 0, stream>>>(
        old_keypoints->LiveVertexConfidence().Ptr(),
        new_keypoints->LiveVertexConfidenceReadOnly().Ptr(),
        old_keypoints->LiveNormalRadius().Ptr(),
        new_keypoints->LiveNormalRadiusReadOnly().Ptr(),
        old_keypoints->ColorTime().Ptr(),
        new_keypoints->ColorTimeReadOnly().Ptr(),
        old_keypoints->SemanticProb().Ptr(),
        new_keypoints->SemanticProbReadOnly().Ptr(),
        old_keypoints->Descriptor().Ptr(),
        new_keypoints->DescriptorReadOnly().Ptr(),
        m_not_matched_indicator.Ptr(),
        m_not_matched_prefix_sum.valid_prefixsum_array.ptr(),
        num_new_kp,
        num_old_kp,
        new_keypoints->DescriptorDim());

    cudaSafeCall(cudaStreamSynchronize(stream));
    old_keypoints->Resize(num_old_kp + *m_host_num_not_matches); // Resize the old keypoints
    std::cout << "KeyPoints appended: " << *m_host_num_not_matches << ", current size is " << new_keypoints->NumKeyPoints() << "." << std::endl;
}