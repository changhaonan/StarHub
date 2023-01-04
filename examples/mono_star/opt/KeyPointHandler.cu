#include "KeyPointHandler.h"
#include <mono_star/common/ConfigParser.h>
#include <mono_star/opt/PenaltyConstants.h>

namespace star::device
{
    constexpr float d_kp_outlier_threshold = 0.03f;

    __global__ void BuildKNNPathKernel(
        const ushortX<d_surfel_knn_size> *__restrict__ kp_knn,
        const floatX<d_surfel_knn_size> *__restrict__ kp_knn_spatial_weight,
        const floatX<d_surfel_knn_size> *__restrict__ kp_knn_connect_weight,
        const int2 *__restrict__ kp_match,
        unsigned short *__restrict__ knn_patch_array,
        float *__restrict__ knn_patch_spatial_weight_array,
        float *__restrict__ knn_patch_connect_weight_array,
        const unsigned num_kp_match)
    {
        const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_kp_match)
            return;

        const int2 match = kp_match[idx];
        const int kp_idx = match.x;
        const unsigned offset = idx * d_surfel_knn_size;
#pragma unroll
        for (auto i = 0; i < d_surfel_knn_size; ++i)
        {
            auto node_knn = kp_knn[kp_idx];
            knn_patch_array[offset + i] = kp_knn[kp_idx][i];
            knn_patch_spatial_weight_array[offset + i] = kp_knn_spatial_weight[kp_idx][i];
            knn_patch_connect_weight_array[offset + i] = kp_knn_connect_weight[kp_idx][i];
        }
    }

    __global__ void UpdateKNNPathDqKernel(
        const DualQuaternion *__restrict__ node_se3,
        const ushortX<d_surfel_knn_size> *__restrict__ kp_knn,
        const int2 *__restrict__ kp_match,
        DualQuaternion *__restrict__ knn_patch_dq_array,
        const unsigned num_kp_match)
    {
        const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_kp_match)
            return;

        const int2 match = kp_match[idx];
        const int kp_idx = match.x;
        const unsigned offset = idx * d_surfel_knn_size;
#pragma unroll
        for (auto i = 0; i < d_surfel_knn_size; ++i)
        {
            auto node_knn = kp_knn[kp_idx];
            knn_patch_dq_array[offset + i] = node_se3[node_knn[i]];
        }
    }

    // Keypoints should be all in world coordinate
    __global__ void ComputeKPJacobianResidual(
        const float4 *__restrict__ kp_vertex_confid_src,
        const float4 *__restrict__ kp_vertex_confid_dst,
        const float4 *__restrict__ kp_normal_radius_src,
        const float4 *__restrict__ kp_normal_radius_dst,
        const int2 *__restrict__ kp_match,
        // KNN structure
        const unsigned short *__restrict__ knn_patch_array,
        const float *__restrict__ knn_patch_spatial_weight_array,
        const float *__restrict__ knn_patch_connect_weight_array,
        const DualQuaternion *__restrict__ node_se3,
        // The output
        GradientOfScalarCost *__restrict__ gradient,
        float *__restrict__ residual,
        const unsigned num_match)
    {
        const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_match)
            return;

        // Prepare
        float term_residual = 0.0f;
        GradientOfScalarCost term_gradient = GradientOfScalarCost();
        int2 match = kp_match[idx];
        const unsigned short *knn_patch_ptr = knn_patch_array + idx * d_surfel_knn_size;
        const float *knn_patch_spatial_weight_ptr = knn_patch_spatial_weight_array + idx * d_surfel_knn_size;
        const float *knn_patch_connect_weight_ptr = knn_patch_connect_weight_array + idx * d_surfel_knn_size;
        DualQuaternion dq_average = averageDualQuaternion(
            node_se3, knn_patch_ptr, knn_patch_spatial_weight_ptr, knn_patch_connect_weight_ptr, d_node_knn_size);
        const mat34 se3 = dq_average.se3_matrix();

        
        // TODO: currently normal is not used
        const float4 target_vertex4 = kp_vertex_confid_dst[match.x]; // The first one is measure
        const float4 target_normal4 = kp_normal_radius_dst[match.x];
        const float3 target_vertex = make_float3(target_vertex4.x, target_vertex4.y, target_vertex4.z);
        const float3 target_normal = make_float3(target_normal4.x, target_normal4.y, target_normal4.z);

        // Warp
        const float4 can_vertex4 = kp_vertex_confid_src[match.y]; // The second one is model
        const float4 can_normal4 = kp_normal_radius_src[match.y];
        const float3 warped_vertex_world = se3.rot * can_vertex4 + se3.trans;
        const float3 warped_normal_world = se3.rot * can_normal4;

        // Compute Jacobian, assume all is valid
        float3 e = warped_vertex_world - target_vertex;
        float e_norm = norm(e);
        float e_norm_inv = 1.f / (e_norm + 1e-8f);
        term_residual = e_norm;
        term_gradient.translation = e * e_norm_inv * 0.5f;
        term_gradient.rotation = cross(warped_vertex_world, e)* e_norm_inv * 0.5f;

        // printf("can: %f, %f, %f, warped: %f %f %f, target: %f %f %f, term: %f.\n",
        //        can_vertex4.x, can_vertex4.y, can_vertex4.z,
        //        warped_vertex_world.x, warped_vertex_world.y, warped_vertex_world.z,
        //        target_vertex.x, target_vertex.y, target_vertex.z, term_residual);

        // Assign to output
        if (e_norm < d_kp_outlier_threshold) {
            residual[idx] = term_residual;
            gradient[idx] = term_gradient;
        }
        else {
            // Regarded as outlier: saturate kernel
            residual[idx] = d_kp_outlier_threshold;
            gradient[idx] = GradientOfScalarCost();
        }
    }
}

star::KeyPointHandler::KeyPointHandler()
{
    AllocateBuffer();
}

star::KeyPointHandler::~KeyPointHandler()
{
    ReleaseBuffer();
}

void star::KeyPointHandler::AllocateBuffer()
{
    m_term_residual.AllocateBuffer(d_max_num_keypoints);
    m_term_gradient.AllocateBuffer(d_max_num_keypoints);

    m_knn_patch_array.AllocateBuffer(d_surfel_knn_size * d_max_num_keypoints);
    m_knn_patch_spatial_weight_array.AllocateBuffer(d_surfel_knn_size * d_max_num_keypoints);
    m_knn_patch_connect_weight_array.AllocateBuffer(d_surfel_knn_size * d_max_num_keypoints);
    m_knn_patch_dq_array.AllocateBuffer(d_surfel_knn_size * d_max_num_keypoints);
}

void star::KeyPointHandler::ReleaseBuffer()
{
    m_term_residual.ReleaseBuffer();
    m_term_gradient.ReleaseBuffer();

    m_knn_patch_array.ReleaseBuffer();
    m_knn_patch_spatial_weight_array.ReleaseBuffer();
    m_knn_patch_connect_weight_array.ReleaseBuffer();
    m_knn_patch_dq_array.ReleaseBuffer();
}

void star::KeyPointHandler::SetInputs(
    const KeyPoint4Solver &keypoint4solver)
{
    m_kp_vertex_confid = keypoint4solver.kp_vertex_confid;
    m_kp_normal_radius = keypoint4solver.kp_normal_radius;

    m_d_kp_vertex_confid = keypoint4solver.d_kp_vertex_confid;
    m_d_kp_normal_radius = keypoint4solver.d_kp_normal_radius;

    m_kp_match = keypoint4solver.kp_match;

    m_kp_knn = keypoint4solver.kp_knn;
    m_kp_knn_spatial_weight = keypoint4solver.kp_knn_spatial_weight;
    m_kp_knn_connect_weight = keypoint4solver.kp_knn_connect_weight;

    m_num_match = m_kp_match.Size();
    ResizeNumMatch(m_num_match);
}

void star::KeyPointHandler::UpdateInputs(
    const GArrayView<DualQuaternion> &node_se3,
    cudaStream_t stream)
{
    if (m_num_match == 0) return;
    // Update node se3
    m_node_se3 = node_se3;
    // Update knn patch dq array
    dim3 blk(128);
    dim3 grid(divUp(m_num_match, blk.x));
    device::UpdateKNNPathDqKernel<<<grid, blk, 0, stream>>>(
        m_node_se3.Ptr(),
        m_kp_knn.Ptr(),
        m_kp_match.Ptr(),
        m_knn_patch_dq_array.Ptr(),
        m_num_match);
}

void star::KeyPointHandler::InitKNNSync(cudaStream_t stream)
{
    if (m_num_match == 0) return;
    // Check valid
    dim3 blk(128);
    dim3 grid(divUp(m_num_match, blk.x));
    device::BuildKNNPathKernel<<<grid, blk, 0, stream>>>(
        m_kp_knn.Ptr(),
        m_kp_knn_spatial_weight.Ptr(),
        m_kp_knn_connect_weight.Ptr(),
        m_kp_match.Ptr(),
        m_knn_patch_array.Ptr(),
        m_knn_patch_spatial_weight_array.Ptr(),
        m_knn_patch_connect_weight_array.Ptr(),
        m_num_match);

    // Sync
    cudaSafeCall(cudaStreamSynchronize(stream));
}

void star::KeyPointHandler::ResizeNumMatch(const unsigned num_match)
{
    m_num_match = num_match;
    m_term_residual.ResizeArrayOrException(num_match);
    m_term_gradient.ResizeArrayOrException(num_match);
    m_knn_patch_array.ResizeArrayOrException(d_surfel_knn_size * num_match);
    m_knn_patch_spatial_weight_array.ResizeArrayOrException(d_surfel_knn_size * num_match);
    m_knn_patch_connect_weight_array.ResizeArrayOrException(d_surfel_knn_size * num_match);
    m_knn_patch_dq_array.ResizeArrayOrException(d_surfel_knn_size * num_match);
}

star::FeatureTerm2Jacobian star::KeyPointHandler::Term2JacobianMap() const
{
    FeatureTerm2Jacobian feature_term2jacobian;
    feature_term2jacobian.knn_patch_array = m_knn_patch_array.View();
    feature_term2jacobian.knn_patch_spatial_weight_array = m_knn_patch_spatial_weight_array.View();
    feature_term2jacobian.knn_patch_connect_weight_array = m_knn_patch_connect_weight_array.View();
    feature_term2jacobian.knn_patch_dq_array = m_knn_patch_dq_array.View();
    feature_term2jacobian.residual_array = m_term_residual.View();
    feature_term2jacobian.gradient_array = m_term_gradient.View();
    return feature_term2jacobian;
}

float star::KeyPointHandler::computeSOR()
{
    auto penalty = PenaltyConstants();

    // Compute keypoint residual and log
    std::vector<float> h_residual;
    m_term_residual.View().Download(h_residual);

    float residual_sum = 0.f;
    for (int i = 0; i < h_residual.size(); ++i)
    {
        residual_sum += h_residual[i] * h_residual[i] * penalty.FeatureSquared();
    }
    std::cout << "SOR [Feature]: " << residual_sum << std::endl;
    return residual_sum;
}

void star::KeyPointHandler::BuildTerm2Jacobian(cudaStream_t stream)
{
    if (m_num_match == 0) return;
    // Compute the term gradient and term residual
    dim3 blk(128);
    dim3 grid(divUp(m_num_match, blk.x));
    device::ComputeKPJacobianResidual<<<grid, blk, 0, stream>>>(
        m_kp_vertex_confid.Ptr(),
        m_d_kp_vertex_confid.Ptr(),
        m_kp_normal_radius.Ptr(),
        m_d_kp_normal_radius.Ptr(),
        m_kp_match.Ptr(),
        // KNN structure
        m_knn_patch_array.Ptr(),
        m_knn_patch_spatial_weight_array.Ptr(),
        m_knn_patch_connect_weight_array.Ptr(),
        m_node_se3.Ptr(),
        // The output
        m_term_gradient.Ptr(),
        m_term_residual.Ptr(),
        m_num_match);
}