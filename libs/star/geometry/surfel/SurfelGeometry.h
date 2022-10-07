#pragma once
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>
#include <star/common/ArrayView.h>
#include <star/common/ArraySlice.h>
#include <star/common/GBufferArray.h>
#include <star/math/device_mat.h>
#include <star/geometry/constants.h>

namespace star
{
	// Forward declaration for class that should have full access to geometry
	struct GLSurfelGeometryVBO;
	class SurfelNodeDeformer;
	class LiveSurfelGeometry;
	class GeometryCompactHandler;

	class SurfelGeometry
	{
	private:
		// The underlying struct for the surfel model
		// Read-Write access, but not owned
		GSliceBufferArray<float4> m_reference_vertex_confid;
		GSliceBufferArray<float4> m_reference_normal_radius;
		GSliceBufferArray<float4> m_live_vertex_confid;
		GSliceBufferArray<float4> m_live_normal_radius;
		GSliceBufferArray<float4> m_color_time;
		GBufferArray<ucharX<d_max_num_semantic>> m_semantic_prob;

		friend struct GLSurfelGeometryVBO;	// map from graphic pipelines
		friend class SurfelNodeDeformer;	// deform the vertex/normal given warp field
		friend class LiveSurfelGeometry;
		friend class GeometryCompactHandler;

		// These are owned
		// KNN structure
		GBufferArray<ushortX<d_surfel_knn_size>> m_surfel_knn;
		GBufferArray<floatX<d_surfel_knn_size>> m_surfel_knn_spatial_weight;
		GBufferArray<floatX<d_surfel_knn_size>> m_surfel_knn_connect_weight;

		// The size recorded for recovering
		size_t m_num_valid_surfels;

	public:
		using Ptr = std::shared_ptr<SurfelGeometry>;
		using ConstPtr = std::shared_ptr<const SurfelGeometry>;
		SurfelGeometry();
		~SurfelGeometry();
		STAR_NO_COPY_ASSIGN_MOVE(SurfelGeometry);

		// Set the valid size of surfels
		size_t NumValidSurfels() const { return m_num_valid_surfels; }
		void ResizeValidSurfelArrays(size_t size);

		// The general access
		GArrayView<float4> ReferenceVertexConfidenceReadOnly() const { return m_reference_vertex_confid.View(); }
		GArrayView<float4> ReferenceNormalRadiusReadOnly() const { return m_reference_normal_radius.View(); }
		GArrayView<float4> LiveVertexConfidenceReadOnly() const { return m_live_vertex_confid.View(); }
		GArrayView<float4> LiveNormalRadiusReadOnly() const { return m_live_normal_radius.View(); }
		GArrayView<float4> ColorTimeReadOnly() const { return m_color_time.View(); }
		GArrayView<ucharX<d_max_num_semantic>> SemanticProbReadOnly() const { return m_semantic_prob.View(); }

		GArraySlice<float4> ReferenceVertexConfidence() { return m_reference_vertex_confid.Slice(); }
		GArraySlice<float4> ReferenceNormalRadius() { return m_reference_normal_radius.Slice(); }
		GArraySlice<float4> LiveVertexConfidence() { return m_live_vertex_confid.Slice(); }
		GArraySlice<float4> LiveNormalRadius() { return m_live_normal_radius.Slice(); }
		GArraySlice<float4> ColorTime() { return m_color_time.Slice(); }
		GArraySlice<ucharX<d_max_num_semantic>> SemanticProb() { return m_semantic_prob.Slice(); }
		
		GArrayView<float4> ReferenceVertexArray() const { return m_reference_vertex_confid.View(); }
		GArrayView<float4> ReferenceNormalArray() const { return m_reference_normal_radius.View(); }

		/* The read-and write access
		 */
		GArraySlice<ushortX<d_surfel_knn_size>> SurfelKNN() { return m_surfel_knn.Slice(); }
		GArraySlice<floatX<d_surfel_knn_size>> SurfelKNNSpatialWeight() { return m_surfel_knn_spatial_weight.Slice(); }
		GArraySlice<floatX<d_surfel_knn_size>> SurfelKNNConnectWeight() { return m_surfel_knn_connect_weight.Slice(); }
		GArrayView<ushortX<d_surfel_knn_size>> SurfelKNNReadOnly() const { return m_surfel_knn.View(); }
		GArrayView<floatX<d_surfel_knn_size>> SurfelKNNSpatialWeightReadOnly() const { return m_surfel_knn_spatial_weight.View(); }
		GArrayView<floatX<d_surfel_knn_size>> SurfelKNNConnectWeightReadOnly() const { return m_surfel_knn_connect_weight.View(); }
		/* The method to for debuging
		 */
		void AddSE3ToVertexAndNormalDebug(const mat34 &se3);

		// Typically for visualization
		struct GeometryAttributes
		{
			GArraySlice<float4> reference_vertex_confid;
			GArraySlice<float4> reference_normal_radius;
			GArraySlice<float4> live_vertex_confid;
			GArraySlice<float4> live_normal_radius;
			GArraySlice<float4> color_time;
			GArraySlice<ucharX<d_max_num_semantic>> semantic_prob;
		};
		GeometryAttributes Geometry();

		/* Static methods
		 */
		// Set tar's ref from src's live. "ReAnchor" the ref geometry
		static void ReAnchor(
			SurfelGeometry::ConstPtr src_geometry,
			SurfelGeometry::Ptr tar_geometry,
			cudaStream_t stream);
	};

	class GLLiveSurfelGeometryVBO;

	// Live geometry; Used for measurement fusion
	class LiveSurfelGeometry
	{

		friend class GLLiveSurfelGeometryVBO; // require full access
		friend class SurfelFilter;

	private:
		GSliceBufferArray<float4> m_live_vertex_confid;
		GSliceBufferArray<float4> m_live_normal_radius;
		GSliceBufferArray<float4> m_color_time;

		// The size recorded for recovering
		size_t m_num_valid_surfels;

	public:
		using Ptr = std::shared_ptr<LiveSurfelGeometry>;
		LiveSurfelGeometry();
		LiveSurfelGeometry(const SurfelGeometry &surfel_geometry);
		~LiveSurfelGeometry();
		STAR_NO_COPY_ASSIGN_MOVE(LiveSurfelGeometry);

		// Set the valid size of surfels
		size_t NumValidSurfels() const { return m_num_valid_surfels; }
		void ResizeValidSurfelArrays(size_t size);

		// Typically for visualization
		struct GeometryAttributes
		{
			GArraySlice<float4> live_vertex_confid;
			GArraySlice<float4> live_normal_radius;
			GArraySlice<float4> color_time;
		};
		GeometryAttributes Geometry() const;

	public:
		// Access
		GArrayView<float4> LiveVertexConfidenceReadOnly() const { return m_live_vertex_confid.View(); }
		GArrayView<float4> ColorTimeReadOnly() const { return m_color_time.View(); }
	};

}
