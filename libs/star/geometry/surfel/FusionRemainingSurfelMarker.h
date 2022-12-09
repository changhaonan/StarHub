#pragma once
#include <star/common/logging.h>
#include <star/common/constants.h>
#include <star/common/ArrayView.h>
#include <star/common/ArraySlice.h>
#include <star/common/GBufferArray.h>
#include <star/common/macro_utils.h>
#include <star/common/algorithm_types.h>
#include <star/math/device_mat.h>
#include <star/geometry/constants.h>
#include <star/geometry/surfel/surfel_fusion_types.h>
#include <star/geometry/surfel/surfel_format.h>
#include <star/geometry/render/Renderer.h>

namespace star
{
	class FusionRemainingSurfelMarker
	{
	public:
		using Ptr = std::shared_ptr<FusionRemainingSurfelMarker>;
		FusionRemainingSurfelMarker(const unsigned num_cam = 1);
		~FusionRemainingSurfelMarker();
		STAR_NO_COPY_ASSIGN_MOVE(FusionRemainingSurfelMarker);
		using FusionMaps = Renderer::FusionMaps;
		using Geometry4Fusion = SurfelGeometry::Geometry4Fusion;
		using Geometry4SemanticFusion = SurfelGeometry::Geometry4SemanticFusion;

		// Set input for STAR version removal
		void SetInputs(
			const Measure4Fusion &measure4fusion,
			const FusionMaps &fusion_maps,
			float current_time,
			const Intrinsic *intrinsic,
			const Extrinsic *cam2world);

		// Set input for SurfelWarp version removal
		void SetInputs(
			const Measure4GeometryRemoval &measure4geometry_removal,
			const Geometry4Fusion &geometry4fusion,
			const FusionMaps &fusion_maps,
			float current_time,
			const Intrinsic *intrinsic,
			const Extrinsic *cam2world);

		// Public API
		Geometry4GeometryRemaining GenerateGeometry4GeometryRemaining() const
		{
			Geometry4GeometryRemaining geometry4geometry_remaining{};
			geometry4geometry_remaining.remaining_indicator = m_remaining_surfel_indicator.View();
			geometry4geometry_remaining.remaining_indicator_prefixsum = m_remaining_indicator_prefixsum.valid_prefixsum_array;
			geometry4geometry_remaining.num_remaining_surfel = (*m_num_remainig_surfel);
			return geometry4geometry_remaining;
		}

		// The processing interface from STAR
		void Initialization(const unsigned surfel_size, cudaStream_t stream);
		void UpdateRemainingSurfelIndicator(cudaStream_t stream);
		void PostProcessRemainingSurfelIndicator(cudaStream_t stream);
		GArrayView<unsigned> GetRemainingSurfelIndicator() const { return m_remaining_surfel_indicator.View(); }
		void RemainingSurfelIndicatorPrefixSumSync(cudaStream_t stream);

		// The processing interface from SurfelWarp
		void InitializationSurfelWarp(const unsigned surfel_size, cudaStream_t stream);
		void UpdateRemainingSurfelIndicatorSurfelWarp(cudaStream_t stream);

		GArrayView<unsigned> GetRemainingSurfelIndicatorPrefixsum() const;
		unsigned GetRemainingSurfelSize() const
		{
			return *m_num_remainig_surfel;
		}
		GArrayView<float> GetRemainingAlignmentError() const { return m_remaining_alignment_error.View(); }

	private:
		Geometry4Fusion m_geometry4fusion;
		Measure4Fusion m_measure4fusion;
		Measure4GeometryRemoval m_measure4geometry_removal;
		FusionMaps m_fusion_maps;

		// The remaining surfel indicator
		GBufferArray<unsigned> m_remaining_surfel_indicator;

		// The camera and time information
		unsigned m_num_cam;
		mat34 m_world2cam[d_max_cam];
		Intrinsic m_intrinsic[d_max_cam];
		float m_current_time;

		PrefixSum m_remaining_indicator_prefixsum;
		unsigned *m_num_remainig_surfel;
		unsigned m_num_valid_surfel;

		// Debug
		GBufferArray<float> m_remaining_alignment_error;
	};

}
