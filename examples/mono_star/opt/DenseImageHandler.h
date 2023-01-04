#pragma once
#include <star/visualization/Visualizer.h>
#include <star/common/logging.h>
#include <star/common/sanity_check.h>
#include <star/common/data_transfer.h>
#include <star/common/common_texture_utils.h>
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>
#include <star/common/ArrayView.h>
#include <star/common/GBufferArray.h>
#include <star/common/surfel_types.h>
#include <star/common/algorithm_types.h>
#include <star/math/DualQuaternion.hpp>
#include <star/opt/solver_types.h>

#include <mono_star/common/Constants.h>
#include <mono_star/common/ConfigParser.h>
#include <mono_star/opt/ImageTermKNNFetcher.h>
#include <memory>

namespace star
{
	/* \brief DenseImageHandler is used to compute twist in depth & optical-flow term.
	 */
	class DenseImageHandler
	{
	private:
		// The info from config
		unsigned m_num_cam;
		unsigned m_image_height[d_max_cam];
		unsigned m_image_width[d_max_cam];
		Intrinsic m_project_intrinsic[d_max_cam];

		// The info from solver; Per-camera
		GArrayView<DualQuaternion> m_node_se3;
		GArrayView2D<KNNAndWeight<d_surfel_knn_size>> m_knn_map[d_max_cam];
		mat34 m_world2cam[d_max_cam];
		mat34 m_cam2world[d_max_cam];

		// The info from depth input
		struct
		{
			cudaTextureObject_t vertex_map[d_max_cam];
			cudaTextureObject_t normal_map[d_max_cam];
		} m_depth_observation;

		// The info from rendered maps
		struct
		{
			cudaTextureObject_t reference_vertex_map[d_max_cam];
			cudaTextureObject_t reference_normal_map[d_max_cam];
			cudaTextureObject_t opticalflow_map[d_max_cam];
			cudaTextureObject_t index_map[d_max_cam];
		} m_geometry_maps;

		// The info from image term fetcher
		ImageTermKNNFetcher::ImageTermPixelAndKNN m_potential_pixels_knn; // Multi-view inherited

	public:
		using Ptr = std::shared_ptr<DenseImageHandler>;
		DenseImageHandler();
		~DenseImageHandler() = default;
		STAR_NO_COPY_ASSIGN_MOVE(DenseImageHandler);

		// Explicit allocate
		void AllocateBuffer();
		void ReleaseBuffer();

		// Set input
		void SetInputs(
			// The potential pixels,
			const ImageTermKNNFetcher::ImageTermPixelAndKNN &pixels_knn,
			// Input from other components
			const GArray2D<KNNAndWeight<d_surfel_knn_size>> *knn_map,
			const Measure4Solver &measure4solver,
			const Render4Solver &render4solver,
			const OpticalFlow4Solver &opticalflow4solver,
			const Extrinsic *world2camera);
		// Update input: se3, connect_weight
		void UpdateInputs(
			const GArrayView<DualQuaternion> &node_se3,
			const ImageTermKNNFetcher::ImageTermPixelAndKNN &pixels_knn);
		// Update the se3
		void UpdateNodeSE3(GArrayView<DualQuaternion> node_se3);

		/* Compute the twist jacobian
		 */
	public:
		void ComputeJacobianTermsFixedIndex(cudaStream_t stream);
		void MergeTerm2Jacobian(cudaStream_t stream);
		DenseImageTerm2Jacobian Term2JacobianMap() const;
		float computeSOR() const; // Debugging method

	private:
		GBufferArray<floatX<d_dense_image_residual_dim>> m_term_residual[d_max_cam];
		GBufferArray<GradientOfDenseImage> m_term_gradient[d_max_cam];
		GBufferArray<floatX<d_dense_image_residual_dim>> m_term_residual_merge;
		GBufferArray<GradientOfDenseImage> m_term_gradient_merge;

		/* Compute the residual map and gather them into nodes. Different from previous residual
		 * The map will return non-zero value at valid pixels that doesnt have corresponded depth pixel
		 * The method is used in Reinit pipeline and visualization.
		 */
	private:
		CudaTextureSurface m_alignment_error_map[d_max_cam];

	public:
		void ComputeAlignmentErrorMapDirect(
			const GArrayView<DualQuaternion> &node_se3, const mat34 *world2camera,
			cudaTextureObject_t *filter_foreground_mask, cudaStream_t stream = 0);
		cudaTextureObject_t GetAlignmentErrorMap(size_t cam_idx) const { return m_alignment_error_map[cam_idx].texture; }

		/* Compute the error and accmulate them on nodes. May distribute them again on
		 * map for further use or visualization
		 */
	private:
		GBufferArray<float> m_node_accumulate_error;
		GBufferArray<float> m_node_accumulate_weight;
		GBufferArray<float> m_node_normalized_error; // Used for output

		// Distribute the node error on maps
		void distributeNodeErrorOnMap(cudaStream_t stream = 0);

	public:
		void ComputeNodewiseError(
			const GArrayView<DualQuaternion> &node_se3,
			const mat34 *world2camera,
			cudaTextureObject_t *filter_foreground_mask,
			cudaStream_t stream = 0);
		void ComputeNodewiseNormalizedError(
			const GArrayView<DualQuaternion> &node_se3,
			const mat34 *world2camera,
			cudaTextureObject_t *filter_foreground_mask,
			const float node_alignment_error_upper_bound,
			cudaStream_t stream = 0);
		void ComputeAlignmentErrorMapFromNode(
			const GArrayView<DualQuaternion> &node_se3, const mat34 *world2camera,
			cudaTextureObject_t *filter_foreground_mask, cudaStream_t stream = 0);

		/* Accessing interface
		 */
	public:
		// The nodewise error
		NodeAlignmentError GetNodeAlignmentError() const
		{
			NodeAlignmentError error;
			error.node_accumulated_error = m_node_accumulate_error.View();
			error.node_accumulate_weight = m_node_accumulate_weight.Ptr();
			return error;
		}
		GArrayView<float> GetNodeNormalizedAlignmentError() const
		{
			return m_node_normalized_error.View();
		}

		/*
		 * \brief: Node outlier check.
		 * The only outlier is that: There is no measurement. But we find an
		 * estimated surface here.
		 */
	public:
		void ComputeNodeOutlier(
			const GArrayView<DualQuaternion> &node_se3,
			const mat34 *world2camera,
			cudaTextureObject_t *filter_foreground_mask,
			cudaStream_t stream = 0);
		GArrayView<unsigned> GetNodeOutlierStatus() const { return m_node_outlier_status.View(); }

	private:
		GBufferArray<unsigned> m_node_outlier_status;

		/* \brief: Sanity checks
		 */
	public:
		// Checking method
		void jacobianTermCheck();
	};
}