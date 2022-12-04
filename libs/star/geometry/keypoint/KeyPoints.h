#pragma once
#include <star/common/common_types_cpu.h>
#include <star/common/ArrayView.h>
#include <star/common/ArraySlice.h>
#include <star/common/GBufferArray.h>
#include <star/geometry/constants.h>

namespace star
{
    class KeyPoints
    {
    public:
        using Ptr = std::shared_ptr<KeyPoints>;
        STAR_NO_COPY_ASSIGN_MOVE(KeyPoints);
        KeyPoints(KeyPointType keypoint_type);
        ~KeyPoints();

        size_t NumKeyPoints() const { return m_num_keypoints; }
        size_t DescriptorDim() const { return m_descriptor_dim; }
        void Resize(size_t size);
        // Fetch API
        GArraySlice<float4> VertexConfid() { return m_vertex_confid.Slice(); }
        GArrayView<float4> VertexConfidReadOnly() const { return m_vertex_confid.View(); }
        GArraySlice<float> Descriptor() { return m_descriptor.Slice(); }
        GArrayView<float> DescriptorReadOnly() const { return m_descriptor.View(); }

    private:
        KeyPointType m_keypoint_type;
        GBufferArray<float4> m_vertex_confid;
        GBufferArray<float> m_descriptor;
        size_t m_num_keypoints;
        size_t m_descriptor_dim;
    };

}