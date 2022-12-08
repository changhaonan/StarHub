#pragma once
#include <star/common/GBufferArray.h>
#include <star/geometry/surfel/SurfelGeometry.h>
#include <star/geometry/constants.h>
#include <star/geometry/surfel/surfel_fusion_types.h>

namespace star
{

	/* \brief Handle many different types of compact
	 */
	class GeometryCompactHandler
	{
	public:
		using Ptr = std::shared_ptr<GeometryCompactHandler>;
		GeometryCompactHandler() : m_num_append(0), m_num_valid_candidate(0), m_num_append_candid(0), m_num_remaining(0), m_num_prev_surfel(0){};
		~GeometryCompactHandler(){};
		STAR_NO_COPY_ASSIGN_MOVE(GeometryCompactHandler);

		void SetInputs(
			SurfelGeometry::Ptr src_geometry,
			SurfelGeometry::Ptr tar_geometry,
			const Geometry4GeometryAppend &geometry4geometry_append,
			const GeometryCandidatePlus &geometry_candidate_plus,
			const Geometry4GeometryRemaining &geometry4geometry_remaining);
		void CompactLiveSurfelToAnotherBufferSync(
			const bool update_semantic,
			cudaStream_t stream);
		void CompactLiveSurfelToAnotherBufferAppendOnlySync(
			const bool update_semantic,
			cudaStream_t stream);
		void CompactLiveSurfelToAnotherBufferRemainingOnlySync(
			const bool update_semantic,
			cudaStream_t stream);

	private:
		SurfelGeometry::Ptr m_src_geometry;
		SurfelGeometry::Ptr m_tar_geometry;
		Geometry4GeometryAppend m_geometry4geometry_append;
		Geometry4GeometryRemaining m_geometry4geometry_remaining;
		GeometryCandidatePlus m_geometry_candidate_plus;
		// Size
		unsigned m_num_append_candid; // Append candidate number
		unsigned m_num_append;		  // Actual append
		unsigned m_num_prev_surfel;	  // Remaining candidate number,
		unsigned m_num_remaining;	  // Actual remaining
		unsigned m_num_valid_candidate;
	};

}