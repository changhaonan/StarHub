#pragma once
#include <star/common/common_types_cpu.h>
#include <star/common/ArrayView.h>
#include <star/common/ArraySlice.h>
#include <star/common/GBufferArray.h>
#include <star/geometry/constants.h>
#include <star/geometry/surfel/SurfelGeometry.h>

namespace star
{
    // Keypoint is based on self-contained surfel geometry
    class KeyPoints : public SurfelGeometrySC
    {
    public:
        using Ptr = std::shared_ptr<KeyPoints>;
        using ConstPtr = std::shared_ptr<const KeyPoints>;
        STAR_NO_COPY_ASSIGN_MOVE(KeyPoints);
        KeyPoints(const KeyPointType keypoint_type);
        ~KeyPoints();

        size_t NumKeyPoints() const { return m_num_valid_surfels; }
        size_t DescriptorDim() const { return m_descriptor_dim; }
        void Resize(size_t size);
        // Fetch API
        GArraySlice<unsigned char> Descriptor() { return m_descriptor.Slice(); }
        GArrayView<unsigned char> DescriptorReadOnly() const { return m_descriptor.View(); }

        // Static API
        static unsigned GetDescriptorDim(KeyPointType keypoint_type)
        {
            if (keypoint_type == KeyPointType::R2D2)
            {
                return 128;
            }
            else if (keypoint_type == KeyPointType::SuperPoints)
            {
                return 256;
            }
            else if (keypoint_type == KeyPointType::ORB) 
            {
                return 32;
            }
            else
            {
                return 0;
            }
        };

        // Static methods
        static void ReAnchor(
            KeyPoints::ConstPtr src_keypoints,
            KeyPoints::Ptr tar_keypoints,
            cudaStream_t stream
        );
    protected:
        KeyPointType m_keypoint_type;
        GBufferArray<unsigned char> m_descriptor;
        size_t m_descriptor_dim;
    };

}