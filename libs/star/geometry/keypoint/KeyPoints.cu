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