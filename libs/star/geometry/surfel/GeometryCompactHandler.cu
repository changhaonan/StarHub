#include "GeometryCompactHandler.h"
#include <device_launch_parameters.h>

namespace star::device
{

	__global__ void CompactSurfelKernel(
		const float4 *__restrict__ prev_live_vertex, // Remaining
		const float4 *__restrict__ prev_live_normal, // Remaining
		const float4 *__restrict__ prev_color_time,
		const ushortX<d_surfel_knn_size> *__restrict__ prev_knn,
		const float4 *__restrict__ append_live_vertex_candid,	   // Candidate
		const float4 *__restrict__ append_live_normal_candid,	   // Candidate
		const float4 *__restrict__ append_color_time_candid,	   // Candidate
		const ushortX<d_surfel_knn_size> *__restrict__ append_knn, // This one is compacted
		const unsigned *__restrict__ remaining_label,
		const unsigned *__restrict__ remaining_prefixsum,
		const unsigned *__restrict__ append_candid_valid_label,
		const unsigned *__restrict__ append_candid_valid_prefixsum,
		float4 *__restrict__ live_vertex_buffer, // Starting from the append position
		float4 *__restrict__ live_normal_buffer,
		float4 *__restrict__ color_time_buffer,
		ushortX<d_surfel_knn_size> *__restrict__ knn_buffer,
		const unsigned num_append_candid, // Append candidate
		const unsigned num_remaining,	  // Remaining in prev
		const unsigned num_prev_surfel,	  // Previous surfel,
		const bool append_only			  // Switch flag
	)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= num_append_candid + num_prev_surfel)
			return;

		if (idx >= num_prev_surfel)
		{ // Append
			if (append_candid_valid_label[idx - num_prev_surfel])
			{
				const auto offset = append_candid_valid_prefixsum[idx - num_prev_surfel] - 1;
				// Candidate
				live_vertex_buffer[num_remaining + offset] = append_live_vertex_candid[idx - num_prev_surfel];
				live_normal_buffer[num_remaining + offset] = append_live_normal_candid[idx - num_prev_surfel];
				color_time_buffer[num_remaining + offset] = append_color_time_candid[idx - num_prev_surfel];
				knn_buffer[num_remaining + offset] = append_knn[idx - num_prev_surfel];
			}
		}
		else
		{
			if (!append_only)
			{
				if (remaining_label[idx])
				{
					const auto offset = remaining_prefixsum[idx] - 1;
					live_vertex_buffer[offset] = prev_live_vertex[idx];
					live_normal_buffer[offset] = prev_live_normal[idx];
					color_time_buffer[offset] = prev_color_time[idx];
					knn_buffer[offset] = prev_knn[idx];
				}
			}
			else
			{ // All remained
				live_vertex_buffer[idx] = prev_live_vertex[idx];
				live_normal_buffer[idx] = prev_live_normal[idx];
				color_time_buffer[idx] = prev_color_time[idx];
				knn_buffer[idx] = prev_knn[idx];
			}
		}
	}

	// With semantic
	__global__ void CompactSurfelKernel(
		const float4 *__restrict__ prev_live_vertex, // Remaining
		const float4 *__restrict__ prev_live_normal, // Remaining
		const float4 *__restrict__ prev_color_time,
		const ucharX<d_max_num_semantic> *__restrict__ prev_semantic_prob,
		const ushortX<d_surfel_knn_size> *__restrict__ prev_knn,
		const float4 *__restrict__ append_live_vertex_candid, // Candidate
		const float4 *__restrict__ append_live_normal_candid, // Candidate
		const float4 *__restrict__ append_color_time_candid,  // Candidate
		const ucharX<d_max_num_semantic> *__restrict__ append_semantic_prob,
		const ushortX<d_surfel_knn_size> *__restrict__ append_knn, // This one is compacted
		const unsigned *__restrict__ remaining_label,
		const unsigned *__restrict__ remaining_prefixsum,
		const unsigned *__restrict__ append_candid_valid_label,
		const unsigned *__restrict__ append_candid_valid_prefixsum,
		float4 *__restrict__ live_vertex_buffer, // Starting from the append position
		float4 *__restrict__ live_normal_buffer,
		float4 *__restrict__ color_time_buffer,
		ucharX<d_max_num_semantic> *__restrict__ semantic_prob_buffer,
		ushortX<d_surfel_knn_size> *__restrict__ knn_buffer,
		const unsigned num_append_candid, // Append candidate
		const unsigned num_remaining,	  // Remaining in prev
		const unsigned num_prev_surfel,	  // Previous surfel,
		const bool append_only			  // Switch flag
	)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= num_append_candid + num_prev_surfel)
			return;

		if (idx >= num_prev_surfel)
		{ // Append
			if (append_candid_valid_label[idx - num_prev_surfel])
			{
				const auto offset = append_candid_valid_prefixsum[idx - num_prev_surfel] - 1;
				// Candidate
				live_vertex_buffer[num_remaining + offset] = append_live_vertex_candid[idx - num_prev_surfel];
				live_normal_buffer[num_remaining + offset] = append_live_normal_candid[idx - num_prev_surfel];
				color_time_buffer[num_remaining + offset] = append_color_time_candid[idx - num_prev_surfel];
				semantic_prob_buffer[num_remaining + offset] = append_semantic_prob[idx - num_prev_surfel];
				knn_buffer[num_remaining + offset] = append_knn[idx - num_prev_surfel];
			}
		}
		else
		{
			if (!append_only)
			{
				if (remaining_label[idx])
				{
					const auto offset = remaining_prefixsum[idx] - 1;
					live_vertex_buffer[offset] = prev_live_vertex[idx];
					live_normal_buffer[offset] = prev_live_normal[idx];
					color_time_buffer[offset] = prev_color_time[idx];
					semantic_prob_buffer[offset] = prev_semantic_prob[idx];
					knn_buffer[offset] = prev_knn[idx];
				}
			}
			else
			{ // All remained
				live_vertex_buffer[idx] = prev_live_vertex[idx];
				live_normal_buffer[idx] = prev_live_normal[idx];
				color_time_buffer[idx] = prev_color_time[idx];
				semantic_prob_buffer[idx] = prev_semantic_prob[idx];
				knn_buffer[idx] = prev_knn[idx];
			}
		}
	}
}

void star::GeometryCompactHandler::SetInputs(
	SurfelGeometry::Ptr src_geometry,
	SurfelGeometry::Ptr tar_geometry,
	const Geometry4GeometryAppend &geometry4geometry_append,
	const GeometryCandidatePlus &geometry_candidate_plus,
	const Geometry4GeometryRemaining &geometry4geometry_remaining)
{
	m_src_geometry = src_geometry;
	m_tar_geometry = tar_geometry;
	m_geometry4geometry_append = geometry4geometry_append;
	m_geometry_candidate_plus = geometry_candidate_plus;
	m_geometry4geometry_remaining = geometry4geometry_remaining;

	// Number
	m_num_append_candid = geometry4geometry_append.num_append_candidate;
	m_num_prev_surfel = src_geometry->NumValidSurfels();
	m_num_remaining = geometry4geometry_remaining.num_remaining_surfel;
	m_num_valid_candidate = geometry_candidate_plus.num_valid_candidate;

	std::cout << "Num append: " << m_num_append_candid
			  << ", Num Prev: " << m_num_prev_surfel
			  << ", Num Remain: " << m_num_remaining
			  << ", Num Valid: " << m_num_valid_candidate
			  << std::endl;

	if (m_num_remaining > m_num_prev_surfel)
	{
		std::cout << "[Warning]: Remaining more than prev size!" << std::endl;
	}
}

void star::GeometryCompactHandler::CompactLiveSurfelToAnotherBufferSync(
	const bool update_semantic,
	cudaStream_t stream)
{
	// 1 - Compact the geometry
	dim3 blk(128);
	dim3 grid(divUp(m_num_prev_surfel + m_num_append_candid, blk.x));
	if (!update_semantic)
	{
		device::CompactSurfelKernel<<<grid, blk, 0, stream>>>(
			m_src_geometry->m_live_vertex_confid.Ptr(),
			m_src_geometry->m_live_normal_radius.Ptr(),
			m_src_geometry->m_color_time.Ptr(),
			m_src_geometry->m_surfel_knn.Ptr(),
			m_geometry4geometry_append.vertex_confid_append_candidate.Ptr(), // candidate
			m_geometry4geometry_append.normal_radius_append_candidate.Ptr(), // candidate
			m_geometry4geometry_append.color_time_append_candidate.Ptr(),	 // candidate
			m_geometry_candidate_plus.append_candidate_surfel_knn.Ptr(),
			m_geometry4geometry_remaining.remaining_indicator.Ptr(),
			m_geometry4geometry_remaining.remaining_indicator_prefixsum.Ptr(),
			m_geometry_candidate_plus.candidate_validity_indicator.Ptr(),
			m_geometry_candidate_plus.candidate_validity_indicator_prefixsum.Ptr(),
			m_tar_geometry->m_live_vertex_confid.Ptr(),
			m_tar_geometry->m_live_normal_radius.Ptr(),
			m_tar_geometry->m_color_time.Ptr(),
			m_tar_geometry->m_surfel_knn.Ptr(),
			m_num_append_candid, // Append candidate number
			m_num_remaining,
			m_num_prev_surfel, // Remaining candidate number
			false);
	}
	else
	{
		device::CompactSurfelKernel<<<grid, blk, 0, stream>>>(
			m_src_geometry->m_live_vertex_confid.Ptr(),
			m_src_geometry->m_live_normal_radius.Ptr(),
			m_src_geometry->m_color_time.Ptr(),
			m_src_geometry->m_semantic_prob.Ptr(),
			m_src_geometry->m_surfel_knn.Ptr(),
			m_geometry4geometry_append.vertex_confid_append_candidate.Ptr(), // candidate
			m_geometry4geometry_append.normal_radius_append_candidate.Ptr(), // candidate
			m_geometry4geometry_append.color_time_append_candidate.Ptr(),	 // candidate
			m_geometry4geometry_append.semantic_prob_append_candidate.Ptr(), // candidate
			m_geometry_candidate_plus.append_candidate_surfel_knn.Ptr(),	 // candidate
			m_geometry4geometry_remaining.remaining_indicator.Ptr(),
			m_geometry4geometry_remaining.remaining_indicator_prefixsum.Ptr(),
			m_geometry_candidate_plus.candidate_validity_indicator.Ptr(),
			m_geometry_candidate_plus.candidate_validity_indicator_prefixsum.Ptr(),
			m_tar_geometry->m_live_vertex_confid.Ptr(),
			m_tar_geometry->m_live_normal_radius.Ptr(),
			m_tar_geometry->m_color_time.Ptr(),
			m_tar_geometry->m_semantic_prob.Ptr(),
			m_tar_geometry->m_surfel_knn.Ptr(),
			m_num_append_candid, // Append candidate number
			m_num_prev_surfel,
			m_num_prev_surfel, // Remaining candidate number
			false);
	}

	cudaSafeCall(cudaStreamSynchronize(stream));

	// 2 - Resize the geometry
	m_tar_geometry->ResizeValidSurfelArrays(m_num_remaining + m_num_valid_candidate);
}

void star::GeometryCompactHandler::CompactLiveSurfelToAnotherBufferAppendOnlySync(
	const bool update_semantic,
	cudaStream_t stream)
{
	// 1. Compact the geometry
	dim3 blk(128);
	dim3 grid(divUp(m_num_prev_surfel + m_num_append_candid, blk.x));
	if (!update_semantic)
	{
		device::CompactSurfelKernel<<<grid, blk, 0, stream>>>(
			m_src_geometry->m_live_vertex_confid.Ptr(),
			m_src_geometry->m_live_normal_radius.Ptr(),
			m_src_geometry->m_color_time.Ptr(),
			m_src_geometry->m_surfel_knn.Ptr(),
			m_geometry4geometry_append.vertex_confid_append_candidate.Ptr(), // candidate
			m_geometry4geometry_append.normal_radius_append_candidate.Ptr(), // candidate
			m_geometry4geometry_append.color_time_append_candidate.Ptr(),	 // candidate
			m_geometry_candidate_plus.append_candidate_surfel_knn.Ptr(),
			nullptr,
			nullptr,
			m_geometry_candidate_plus.candidate_validity_indicator.Ptr(),
			m_geometry_candidate_plus.candidate_validity_indicator_prefixsum.Ptr(),
			m_tar_geometry->m_live_vertex_confid.Ptr(),
			m_tar_geometry->m_live_normal_radius.Ptr(),
			m_tar_geometry->m_color_time.Ptr(),
			m_tar_geometry->m_surfel_knn.Ptr(),
			m_num_append_candid, // Append candidate number
			m_num_prev_surfel,
			m_num_prev_surfel, // Remaining candidate number
			true);
	}
	else
	{
		device::CompactSurfelKernel<<<grid, blk, 0, stream>>>(
			m_src_geometry->m_live_vertex_confid.Ptr(),
			m_src_geometry->m_live_normal_radius.Ptr(),
			m_src_geometry->m_color_time.Ptr(),
			m_src_geometry->m_semantic_prob.Ptr(),
			m_src_geometry->m_surfel_knn.Ptr(),
			m_geometry4geometry_append.vertex_confid_append_candidate.Ptr(), // candidate
			m_geometry4geometry_append.normal_radius_append_candidate.Ptr(), // candidate
			m_geometry4geometry_append.color_time_append_candidate.Ptr(),	 // candidate
			m_geometry4geometry_append.semantic_prob_append_candidate.Ptr(), // candidate
			m_geometry_candidate_plus.append_candidate_surfel_knn.Ptr(),	 // candidate
			nullptr,
			nullptr,
			m_geometry_candidate_plus.candidate_validity_indicator.Ptr(),
			m_geometry_candidate_plus.candidate_validity_indicator_prefixsum.Ptr(),
			m_tar_geometry->m_live_vertex_confid.Ptr(),
			m_tar_geometry->m_live_normal_radius.Ptr(),
			m_tar_geometry->m_color_time.Ptr(),
			m_tar_geometry->m_semantic_prob.Ptr(),
			m_tar_geometry->m_surfel_knn.Ptr(),
			m_num_append_candid, // Append candidate number
			m_num_prev_surfel,
			m_num_prev_surfel, // Remaining candidate number
			true);
	}

	cudaSafeCall(cudaStreamSynchronize(stream));

	// 2. Resize the geometry
	m_tar_geometry->ResizeValidSurfelArrays(m_num_prev_surfel + m_num_valid_candidate);

	// 3. Log info
	std::cout << "[Info] Append Only; Geometry resize to " << m_num_prev_surfel + m_num_valid_candidate << std::endl;
}

void star::GeometryCompactHandler::CompactLiveSurfelToAnotherBufferRemainingOnlySync(
	const bool update_semantic,
	cudaStream_t stream)
{
	// 1. Compact the geometry
	dim3 blk(128);
	dim3 grid(divUp(m_num_prev_surfel + m_num_append_candid, blk.x));
	if (!update_semantic)
	{
		device::CompactSurfelKernel<<<grid, blk, 0, stream>>>(
			m_src_geometry->m_live_vertex_confid.Ptr(),
			m_src_geometry->m_live_normal_radius.Ptr(),
			m_src_geometry->m_color_time.Ptr(),
			m_src_geometry->m_surfel_knn.Ptr(),
			nullptr, // candidate
			nullptr, // candidate
			nullptr, // candidate
			nullptr,
			m_geometry4geometry_remaining.remaining_indicator.Ptr(),
			m_geometry4geometry_remaining.remaining_indicator_prefixsum.Ptr(),
			nullptr,
			nullptr,
			m_tar_geometry->m_live_vertex_confid.Ptr(),
			m_tar_geometry->m_live_normal_radius.Ptr(),
			m_tar_geometry->m_color_time.Ptr(),
			m_tar_geometry->m_surfel_knn.Ptr(),
			m_num_append_candid, // Append candidate number
			m_num_remaining,
			m_num_prev_surfel, // Remaining candidate number
			false);
	}
	else
	{
		device::CompactSurfelKernel<<<grid, blk, 0, stream>>>(
			m_src_geometry->m_live_vertex_confid.Ptr(),
			m_src_geometry->m_live_normal_radius.Ptr(),
			m_src_geometry->m_color_time.Ptr(),
			m_src_geometry->m_semantic_prob.Ptr(),
			m_src_geometry->m_surfel_knn.Ptr(),
			nullptr, // candidate
			nullptr, // candidate
			nullptr, // candidate
			nullptr, // candidate
			nullptr, // candidate
			m_geometry4geometry_remaining.remaining_indicator.Ptr(),
			m_geometry4geometry_remaining.remaining_indicator_prefixsum.Ptr(),
			nullptr,
			nullptr,
			m_tar_geometry->m_live_vertex_confid.Ptr(),
			m_tar_geometry->m_live_normal_radius.Ptr(),
			m_tar_geometry->m_color_time.Ptr(),
			m_tar_geometry->m_semantic_prob.Ptr(),
			m_tar_geometry->m_surfel_knn.Ptr(),
			m_num_append_candid, // Append candidate number
			m_num_prev_surfel,
			m_num_prev_surfel, // Remaining candidate number
			false);
	}

	cudaSafeCall(cudaStreamSynchronize(stream));

	// 2. Resize the geometry
	m_tar_geometry->ResizeValidSurfelArrays(m_num_remaining);

	// 3. Log info
	std::cout << "[Info] Remaining-only; Geometry resize to " << m_num_remaining << std::endl;
}