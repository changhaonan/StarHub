#pragma once
#include <star/common/common_types.h>
#include <star/common/common_utils.h>
#include <star/common/constants.h>
#include <star/common/ArraySlice.h>
#include <star/common/ArrayView.h>
#include <star/common/GBufferArray.h>
#include <star/common/algorithm_types.h>
#include <star/geometry/constants.h>
#include <star/geometry/surfel/surfel_fusion_types.h>

namespace star
{
	class DynamicGeometryAppendHandler
	{
	public:
		using Ptr = std::shared_ptr<DynamicGeometryAppendHandler>;
		DynamicGeometryAppendHandler();
		~DynamicGeometryAppendHandler();
		STAR_NO_COPY_ASSIGN_MOVE(DynamicGeometryAppendHandler);

		void SetInputs(
			const GArrayView<float4> &append_candidate_surfel);
		void PrefixSumSync(cudaStream_t stream);
		void Initialize(cudaStream_t stream);

		// Public API
		GArraySlice<ushortX<d_surfel_knn_size>> AppendSurfelKnn()
		{
			return m_append_surfel_knn.Slice();
		}
		GeometryCandidateIndicator GenerateGeometryCandidateIndicator()
		{
			GeometryCandidateIndicator geometry_candidate_indicator;
			geometry_candidate_indicator.candidate_validity_indicator = m_candidate_validity_indicator.Slice();
			geometry_candidate_indicator.candidate_unsupported_indicator = m_candidate_unsupported_indicator.Slice();
			return geometry_candidate_indicator;
		}
		GeometryCandidatePlus GenerateGeometryCandidatePlus() const
		{
			GeometryCandidatePlus geometry_candidate_plus;
			geometry_candidate_plus.candidate_validity_indicator = m_candidate_validity_indicator.View();
			geometry_candidate_plus.candidate_validity_indicator_prefixsum = m_candidate_validity_indicator_prefixsum.valid_prefixsum_array;
			geometry_candidate_plus.append_candidate_surfel_knn = m_append_surfel_knn.View();
			geometry_candidate_plus.num_supported_candidate = *m_num_unsupported_candidate;
			geometry_candidate_plus.num_valid_candidate = *m_num_valid_candidate;
			return geometry_candidate_plus;
		}
		void ComputeUnsupportedCandidate(cudaStream_t stream);
		GArrayView<float4> GenerateUnsupportedCandidate() const { return m_unsupported_candidate_surfel.View(); }

	private:
		// Indicator
		unsigned m_num_candidate;
		unsigned *m_num_valid_candidate;
		unsigned *m_num_unsupported_candidate;
		GBufferArray<unsigned> m_candidate_validity_indicator;
		GBufferArray<unsigned> m_candidate_unsupported_indicator;
		PrefixSum m_candidate_validity_indicator_prefixsum;
		PrefixSum m_candidate_unsupported_indicator_prefixsum;
		// Append candidate
		GArrayView<float4> m_append_candidate_surfel;
		GBufferArray<float4> m_unsupported_candidate_surfel;

		// AppendGeometry
		GBufferArray<ushortX<d_surfel_knn_size>> m_append_surfel_knn;
	};

}