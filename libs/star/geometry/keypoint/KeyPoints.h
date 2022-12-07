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
        STAR_NO_COPY_ASSIGN_MOVE(KeyPoints);
        KeyPoints(KeyPointType keypoint_type);
        ~KeyPoints();

        size_t NumKeyPoints() const { return m_num_valid_surfels; }
        size_t DescriptorDim() const { return m_descriptor_dim; }
        void Resize(size_t size);
        // Fetch API
        GArraySlice<float> Descriptor() { return m_descriptor.Slice(); }
        GArrayView<float> DescriptorReadOnly() const { return m_descriptor.View(); }
    protected:
        KeyPointType m_keypoint_type;
        GBufferArray<float4> m_vertex_confid;
        GBufferArray<float4> m_normal_radius;
        GBufferArray<float4> m_color_time;
        GBufferArray<float> m_descriptor;
        size_t m_descriptor_dim;
    };

}