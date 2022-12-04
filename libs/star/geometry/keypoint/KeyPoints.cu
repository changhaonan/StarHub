#include "KeyPoints.h"

star::KeyPoints::KeyPoints(KeyPointType keypoint_type) : m_keypoint_type(keypoint_type)
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

    m_vertex_confid.AllocateBuffer(d_max_num_keypoints);
    m_descriptor.AllocateBuffer(d_max_num_keypoints * m_descriptor_dim);
}

star::KeyPoints::~KeyPoints()
{
    m_vertex_confid.ReleaseBuffer();
    m_descriptor.ReleaseBuffer();
}

void star::KeyPoints::Resize(size_t size)
{
    m_vertex_confid.ResizeArrayOrException(size);
    m_descriptor.ResizeArrayOrException(size * m_descriptor_dim);
}
