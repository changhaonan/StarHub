#include "KeyPoints.h"

star::KeyPoints::KeyPoints(const KeyPointType keypoint_type)
    : SurfelGeometrySC(), m_keypoint_type(keypoint_type)
{
    switch (keypoint_type)
    {
    case KeyPointType::R2D2:
        m_descriptor_dim = 128;
        break;
    case KeyPointType::SuperPoints:
        m_descriptor_dim = 256;
        break;
    default:
        break;
    }

    m_descriptor.AllocateBuffer(d_max_num_keypoints * m_descriptor_dim);
}

star::KeyPoints::~KeyPoints()
{
    m_descriptor.ReleaseBuffer();
}

void star::KeyPoints::Resize(size_t size)
{
    ResizeValidSurfelArrays(size);
    m_descriptor.ResizeArrayOrException(size * m_descriptor_dim);
}

void star::KeyPoints::ReAnchor(
    KeyPoints::ConstPtr src_keypoints,
    KeyPoints::Ptr tar_keypoints,
    cudaStream_t stream)
{
    // Copy owned data
    cudaSafeCall(
        cudaMemcpyAsync(
            tar_keypoints->SurfelKNN(),
            src_keypoints->SurfelKNNReadOnly(),
            src_keypoints->SurfelKNNReadOnly().ByteSize(),
            cudaMemcpyDeviceToDevice, stream));
    cudaSafeCall(
        cudaMemcpyAsync(
            tar_keypoints->SurfelKNNSpatialWeight(),
            src_keypoints->SurfelKNNSpatialWeightReadOnly(),
            src_keypoints->SurfelKNNSpatialWeightReadOnly().ByteSize(),
            cudaMemcpyDeviceToDevice, stream));
    cudaSafeCall(
        cudaMemcpyAsync(
            tar_keypoints->SurfelKNNConnectWeight(),
            src_keypoints->SurfelKNNConnectWeightReadOnly(),
            src_keypoints->SurfelKNNConnectWeightReadOnly().ByteSize(),
            cudaMemcpyDeviceToDevice, stream));
    // Copy geometry data
    cudaSafeCall(
        cudaMemcpyAsync(
            tar_keypoints->ReferenceVertexConfidence(),
            src_keypoints->LiveVertexConfidenceReadOnly(),
            src_keypoints->LiveVertexConfidenceReadOnly().ByteSize(),
            cudaMemcpyDeviceToDevice, stream));
    cudaSafeCall(
        cudaMemcpyAsync(
            tar_keypoints->ReferenceNormalRadius(),
            src_keypoints->LiveNormalRadiusReadOnly(),
            src_keypoints->LiveNormalRadiusReadOnly().ByteSize(),
            cudaMemcpyDeviceToDevice, stream));
    cudaSafeCall(
        cudaMemcpyAsync(
            tar_keypoints->ColorTime(),
            src_keypoints->ColorTimeReadOnly(),
            src_keypoints->ColorTimeReadOnly().ByteSize(),
            cudaMemcpyDeviceToDevice, stream));
    // Optional
    cudaSafeCall(
        cudaMemcpyAsync(
            tar_keypoints->SemanticProb(),
            src_keypoints->SemanticProbReadOnly(),
            src_keypoints->SemanticProbReadOnly().ByteSize(),
            cudaMemcpyDeviceToDevice, stream));
    // Descriptor
    cudaSafeCall(
        cudaMemcpyAsync(
            tar_keypoints->Descriptor(),
            src_keypoints->DescriptorReadOnly(),
            src_keypoints->DescriptorReadOnly().ByteSize(),
            cudaMemcpyDeviceToDevice, stream));

    // Sync & Resize
    cudaSafeCall(cudaStreamSynchronize(stream));
    tar_keypoints->Resize(src_keypoints->NumKeyPoints());
}