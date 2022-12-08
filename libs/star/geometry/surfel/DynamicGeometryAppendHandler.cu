#include <device_launch_parameters.h>
#include "DynamicGeometryAppendHandler.h"

namespace star::device
{
	__global__ void GenerateUnsupportCandidateKernel(
		const float4 *__restrict__ candidate_surfel,
		const unsigned *__restrict__ support_indicator,
		const unsigned *__restrict__ support_indicator_offset,
		float4 *unsupported_candidate_surfel,
		const unsigned candidate_size)
	{
		auto idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= candidate_size)
			return;

		if (support_indicator[idx])
		{
			auto offset = support_indicator_offset[idx];
			unsupported_candidate_surfel[offset] = candidate_surfel[idx];
		}
	}

}

star::DynamicGeometryAppendHandler::DynamicGeometryAppendHandler() : m_num_candidate(0)
{
	// Indicator
	m_candidate_validity_indicator.AllocateBuffer(d_max_num_surfels);
	m_candidate_unsupported_indicator.AllocateBuffer(d_max_num_surfels);
	m_candidate_validity_indicator_prefixsum.AllocateBuffer(d_max_num_surfels);
	m_candidate_unsupported_indicator_prefixsum.AllocateBuffer(d_max_num_surfels);
	m_unsupported_candidate_surfel.AllocateBuffer(d_max_num_surfels);
	m_append_surfel_knn.AllocateBuffer(d_max_num_surfels);

	// Counter
	cudaSafeCall(cudaMallocHost((void **)&m_num_unsupported_candidate, sizeof(unsigned)));
	cudaSafeCall(cudaMallocHost((void **)&m_num_valid_candidate, sizeof(unsigned)));
}

star::DynamicGeometryAppendHandler::~DynamicGeometryAppendHandler()
{
	m_candidate_validity_indicator.ReleaseBuffer();
	m_candidate_unsupported_indicator.ReleaseBuffer();
	m_append_surfel_knn.ReleaseBuffer();
	m_unsupported_candidate_surfel.ReleaseBuffer();

	cudaSafeCall(cudaFreeHost(m_num_unsupported_candidate));
	cudaSafeCall(cudaFreeHost(m_num_valid_candidate));
}

void star::DynamicGeometryAppendHandler::SetInputs(
	const GArrayView<float4> &append_candidate_surfel)
{
	m_num_candidate = append_candidate_surfel.Size();
	m_append_candidate_surfel = append_candidate_surfel;
}

void star::DynamicGeometryAppendHandler::PrefixSumSync(cudaStream_t stream)
{
	m_candidate_validity_indicator_prefixsum.InclusiveSum(m_candidate_validity_indicator.View(), stream);
	// Query the size
	cudaSafeCall(cudaMemcpyAsync(
		m_num_valid_candidate,
		m_candidate_validity_indicator_prefixsum.valid_prefixsum_array.ptr() + m_candidate_validity_indicator_prefixsum.valid_prefixsum_array.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream));

	m_candidate_unsupported_indicator_prefixsum.InclusiveSum(m_candidate_unsupported_indicator.View(), stream);
	// Query the size
	cudaSafeCall(cudaMemcpyAsync(
		m_num_unsupported_candidate,
		m_candidate_unsupported_indicator_prefixsum.valid_prefixsum_array.ptr() + m_candidate_unsupported_indicator_prefixsum.valid_prefixsum_array.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream));
	cudaSafeCall(cudaStreamSynchronize(stream));
	std::cout << "Num unsupported candidate: " << *m_num_unsupported_candidate << std::endl;
}

void star::DynamicGeometryAppendHandler::Initialize(cudaStream_t stream)
{
	m_candidate_validity_indicator.ResizeArrayOrException(m_num_candidate);
	m_candidate_unsupported_indicator.ResizeArrayOrException(m_num_candidate);
	m_append_surfel_knn.ResizeArrayOrException(m_num_candidate);

	cudaSafeCall(cudaMemsetAsync(
		m_candidate_validity_indicator.Ptr(), 0, m_candidate_validity_indicator.ArrayByteSize(), stream));
	cudaSafeCall(cudaMemsetAsync(
		m_candidate_unsupported_indicator.Ptr(), 0, m_candidate_unsupported_indicator.ArrayByteSize(), stream));
	cudaSafeCall(cudaMemsetAsync(
		m_append_surfel_knn.Ptr(), 0, m_append_surfel_knn.ArrayByteSize(), stream));
}

void star::DynamicGeometryAppendHandler::ComputeUnsupportedCandidate(cudaStream_t stream)
{
	dim3 blk(128);
	dim3 grid(divUp(m_num_candidate, blk.x));
	device::GenerateUnsupportCandidateKernel<<<grid, blk, 0, stream>>>(
		m_append_candidate_surfel.Ptr(),
		m_candidate_unsupported_indicator.Ptr(),
		m_candidate_unsupported_indicator_prefixsum.valid_prefixsum_array.ptr(),
		m_unsupported_candidate_surfel.Ptr(),
		m_num_candidate);
	cudaSafeCall(cudaStreamSynchronize(stream));
	m_unsupported_candidate_surfel.ResizeArrayOrException(*m_num_unsupported_candidate);
}
