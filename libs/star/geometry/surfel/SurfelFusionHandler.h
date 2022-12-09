#pragma once
#include <star/common/common_types.h>
#include <star/common/common_utils.h>
#include <star/common/constants.h>
#include <star/common/ArraySlice.h>
#include <star/common/ArrayView.h>
#include <star/common/GBufferArray.h>
#include <star/common/algorithm_types.h>
#include <star/geometry/constants.h>
#include <star/geometry/render/Renderer.h>
#include <star/geometry/surfel/surfel_fusion_types.h>

namespace star
{
	// The task of surfel fusion handler is: given the fusion
	// geometry and the fusion maps, fuse it to existing geometry
	// and compute indicators for which surfels should remain and
	// which depth pixel will potentiall be appended to the surfel array
	// This kernel parallelize over images (not the surfel array)
	// Can also handle surfel append
	class SurfelFusionHandler
	{
	public:
		using Ptr = std::shared_ptr<SurfelFusionHandler>;
		SurfelFusionHandler(
			const unsigned num_cam,
			const unsigned *img_cols,
			const unsigned *img_rows,
			const bool enable_semantic_surfel);
		~SurfelFusionHandler();
		STAR_NO_COPY_ASSIGN(SurfelFusionHandler);
		using FusionMaps = Renderer::FusionMaps;
		using Geometry4Fusion = SurfelGeometry::Geometry4Fusion;
		using Geometry4SemanticFusion = SurfelGeometry::Geometry4SemanticFusion;

		// The input requires all CamearObservation
		void SetInputs(
			const FusionMaps &fusion_maps,
			const Measure4Fusion &measure4fusion,
			Geometry4Fusion &geometry4fusion,
			float current_time,
			const Extrinsic *cam2world);
		void SetInputs(
			const FusionMaps &fusion_maps,
			const Measure4Fusion &measure4fusion,
			const Segmentation4SemanticFusion &segmentation4semantic_fusion,
			Geometry4Fusion &geometry4fusion,
			Geometry4SemanticFusion &geometry4semantic_fusion,
			float current_time,
			const Extrinsic *cam2world);

		/*
		 * Fusion:
		 * 1. Update matched surfels.
		 * 2. Unmatched, if far from exisiting, will be marked as candidates.
		 */
		void ProcessFusion(const bool update_semantic, cudaStream_t stream = 0);
		void CompactAppendedCandidate(
			const GArrayView<float4> &vertex_config_array,
			const GArrayView<float4> &normal_radius_array,
			const GArrayView<float4> &color_time_array,
			cudaStream_t stream = 0);
		void CompactAppendedCandidate(
			const GArrayView<float4> &vertex_config_array,
			const GArrayView<float4> &normal_radius_array,
			const GArrayView<float4> &color_time_array,
			const GArrayView<ucharX<d_max_num_semantic>> &semantic_prob_array,
			cudaStream_t stream = 0);

		/*
		 * Fusion with re-init:
		 * 1. List all unmatched as appending.
		 * 2. Update all matched with measurement color.
		 */
		void ProcessFusionReinit(cudaStream_t stream = 0);

		void ZeroInitializeIndicator(
			const unsigned num_geometry_surfels, const unsigned num_observation_surfels, cudaStream_t stream = 0);

		/* Public API
		 */
		// Vertex are in the live frame
		GArraySlice<unsigned> GetRemainingSurfelIndicator();
		GArrayView<float4> GetCompactAppendedVertex() const
		{
			return m_appended_surfel_vertex.View();
		}
		GArrayView<float4> GetCompactAppendedNormal() const
		{
			return m_appended_surfel_normal.View();
		}
		GArrayView<float4> GetCompactAppendedColorTime() const
		{
			return m_appended_surfel_color_time.View();
		}
		GArrayView<ucharX<d_max_num_semantic>> GetCompactAppendedSemanticProb() const
		{
			return m_appended_surfel_semantic_prob.View();
		}
		unsigned GetNumAppendSurfelCandid() const { return m_num_appended_surfel; }
		Geometry4GeometryAppend GenerateGeometry4GeometryAppend() const
		{
			Geometry4GeometryAppend geometry4geometry_append;
			geometry4geometry_append.vertex_confid_append_candidate = m_appended_surfel_vertex.View();
			geometry4geometry_append.normal_radius_append_candidate = m_appended_surfel_normal.View();
			geometry4geometry_append.color_time_append_candidate = m_appended_surfel_color_time.View();
			geometry4geometry_append.semantic_prob_append_candidate = m_appended_surfel_semantic_prob.View();
			geometry4geometry_append.num_append_candidate = m_appended_surfel_vertex.ArraySize();
			return geometry4geometry_append;
		}
		/*
		 * Process data fusion using compaction
		 */
	private:
		void prepareFuserArguments(const bool update_semantic, const unsigned cam_idx, void *fuser_ptr);
		void processFusionAndAppendLabel(const bool update_semantic, cudaStream_t stream = 0); // Labeling append & fused
		void processFusionReinit(const bool update_semantic, cudaStream_t stream = 0);

		GBufferArray<float4> m_appended_surfel_vertex;
		GBufferArray<float4> m_appended_surfel_normal;
		GBufferArray<float4> m_appended_surfel_color_time;
		GBufferArray<ucharX<d_max_num_semantic>> m_appended_surfel_semantic_prob; // Optional
		// Indicator
		unsigned m_num_appended_surfel;
		PrefixSum m_appended_surfel_indicator_prefixsum;
		GBufferArray<unsigned> m_appended_surfel_indicator;

		// Basic parameters
		unsigned m_num_cam;
		unsigned m_image_rows[d_max_cam];
		unsigned m_image_cols[d_max_cam];

		// The input from outside
		FusionMaps m_fusion_maps;
		Geometry4Fusion m_geometry4fusion;
		Measure4Fusion m_measure4fusion;
		// Optional
		Segmentation4SemanticFusion m_segmentation4semantic_fusion;
		Geometry4SemanticFusion m_geometry4semantic_fusion;
		bool m_enable_semantic_surfel;

		float m_current_time;
		mat34 m_world2cam[d_max_cam];

		// The buffer maintained by this class
		GBufferArray<unsigned> m_remaining_surfel_indicator;
	};
}