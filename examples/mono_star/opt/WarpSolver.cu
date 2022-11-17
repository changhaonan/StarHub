#include "star/warp_solver/WarpSolver.h"
#include <device_launch_parameters.h>
#include "visualization/Visualizer.h"

namespace star {namespace device {

	__global__ void queryPixelKNNKernel(
		cudaTextureObject_t index_map,
		const ushortX<d_surfel_knn_size>* __restrict__ surfel_knn,
		const floatX<d_surfel_knn_size>* __restrict__  surfel_knn_spatial_weight,
		const floatX<d_surfel_knn_size>* __restrict__  surfel_knn_connect_weight,
		PtrStepSz<KNNAndWeight<d_surfel_knn_size>> knn_map
	) {
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x < knn_map.cols && y < knn_map.rows)
		{
			KNNAndWeight<d_surfel_knn_size> knn_weight;
			knn_weight.set_invalid();
			const unsigned index = tex2D<unsigned>(index_map, x, y);
			if (index != 0xFFFFFFFF) {
				knn_weight.knn = surfel_knn[index];
				knn_weight.spatial_weight = surfel_knn_spatial_weight[index];
				knn_weight.connect_weight = surfel_knn_connect_weight[index];
			}
			knn_map.ptr(y)[x] = knn_weight;

#ifdef OPT_DEBUG_CHECK
			if (index != 0xFFFFFFFF) {
				bool flag_zero_connect = true;
				for (auto i = 0; i < d_surfel_knn_size; ++i) {
					if (knn_weight.connect_weight[i] != 0.f) {
						flag_zero_connect = false;
					}
				}
				if (flag_zero_connect) {
					printf("QueryKNN: Zero connect at surfel %d.\n", index);
				}
			}
#endif // OPT_DEBUG_CHECK

		}
	}

}
}

void star::WarpSolver::QueryPixelKNN(cudaStream_t stream) {
	dim3 blk(16, 16);
	dim3 grid;
	for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx) {  // Per-camera
		grid.x = divUp(m_image_width[cam_idx], blk.x);
		grid.y = divUp(m_image_height[cam_idx], blk.y);
		device::queryPixelKNNKernel<<<grid, blk, 0, stream>>>(
			m_render4solver.index_map[cam_idx],
			m_geometry4solver.surfel_knn.Ptr(),
			m_geometry4solver.surfel_knn_spatial_weight.Ptr(),
			m_geometry4solver.surfel_knn_connect_weight.Ptr(),
			m_knn_map[cam_idx]);
	}

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void star::WarpSolver::allocatePCGSolverBuffer() {
	const auto max_matrix_size = d_node_variable_dim * Constants::kMaxNumNodes;
	m_pcg_solver = std::make_shared<BlockPCG<d_node_variable_dim>>(max_matrix_size);
}

void star::WarpSolver::releasePCGSolverBuffer() {
}