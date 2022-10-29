// #pragma once
// #include <star/common/common_types.h>
// #include "global_configs.h"
// #include "Constants.h"

// namespace star {

// 	struct FusionMaps {
// 		using Ptr = std::shared_ptr<FusionMaps>;
// 		cudaTextureObject_t warp_vertex_confid_map[d_max_cam] = { 0 };
// 		cudaTextureObject_t warp_normal_radius_map[d_max_cam] = { 0 };
// 		cudaTextureObject_t index_map[d_max_cam] = { 0 };
// 		cudaTextureObject_t color_time_map[d_max_cam] = { 0 };
// 	};
	
// 	struct Geometry4Fusion {
// 		GArraySlice<float4> vertex_confid;
// 		GArraySlice<float4> normal_radius;
// 		GArraySlice<float4> color_time;
// 		unsigned num_valid_surfel = 0;
// 	};

// 	struct Measure4Fusion {
// 		cudaTextureObject_t vertex_confid_map[d_max_cam] = { 0 };
// 		cudaTextureObject_t normal_radius_map[d_max_cam] = { 0 };
// 		cudaTextureObject_t color_time_map[d_max_cam] = { 0 };
// 		cudaTextureObject_t index_map[d_max_cam] = { 0 };
// 		unsigned num_valid_surfel = 0;
// 	};

// 	/* Only used for geometry removal, so don't need too much
// 	*/
// 	struct Measure4GeometryRemoval {
// 		cudaTextureObject_t depth4removal_map[d_max_cam] = { 0 };
// 	};
	
// 	struct Geometry4SemanticFusion {
// 		GArraySlice<ucharX<d_max_num_semantic>> semantic_prob;
// 		unsigned num_valid_surfel = 0;
// 	};

// 	struct Segmentation4SemanticFusion {
// 		cudaTextureObject_t segmentation[d_max_cam] = { 0 };
// 	};

// 	/* For Geometry Add
// 	*/
// 	struct GeometryCandidateIndicator {
// 		GArraySlice<unsigned> candidate_validity_indicator;
// 		GArraySlice<unsigned> candidate_unsupported_indicator;
// 	};

// 	// Geometry Add, but additional info
// 	struct GeometryCandidatePlus {
// 		GArrayView<unsigned> candidate_validity_indicator;
// 		GArrayView<unsigned> candidate_validity_indicator_prefixsum;
// 		GArrayView<ushortX<d_surfel_knn_size>> append_candidate_surfel_knn;
// 		unsigned num_valid_candidate = 0;
// 		unsigned num_supported_candidate = 0;
// 	};
	
// 	struct Geometry4GeometryAppend {
// 		// Candidate
// 		GArrayView<float4> vertex_confid_append_candidate;
// 		GArrayView<float4> normal_radius_append_candidate;
// 		GArrayView<float4> color_time_append_candidate;
// 		GArrayView<ucharX<d_max_num_semantic>> semantic_prob_append_candidate;  // (Optional) one
// 		unsigned num_append_candidate = 0;
// 	};

// 	struct Geometry4GeometryRemaining {
// 		GArrayView<unsigned> remaining_indicator;
// 		GArrayView<unsigned> remaining_indicator_prefixsum;
// 		unsigned num_remaining_surfel = 0;
// 	};
	
// }