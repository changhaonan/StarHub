#pragma once
#include <mono_star/common/ConfigParser.h>
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>
#include <star/common/algorithm_types.h>
#include <star/common/ArrayView.h>
#include <star/common/GBufferArray.h>
#include <star/common/surfel_types.h>
#include <star/opt/solver_types.h>
#include <curand.h>
#include <memory>

namespace star
{
	class ImageTermKNNFetcher
	{
	public:
		// The contructor group
		using Ptr = std::shared_ptr<ImageTermKNNFetcher>;
		ImageTermKNNFetcher();
		~ImageTermKNNFetcher();
		STAR_NO_COPY_ASSIGN_MOVE(ImageTermKNNFetcher);

		// The input from the solver
		void SetInputs(
			const unsigned num_cam,
			const GArrayView2D<KNNAndWeight<d_surfel_knn_size>> *knn_patch_map,
			cudaTextureObject_t *index_map,
			cudaTextureObject_t *opticalflow_map);
		void SetInputs(
			size_t cam_idx,
			const GArrayView2D<KNNAndWeight<d_surfel_knn_size>> knn_patch_map,
			cudaTextureObject_t index_map,
			cudaTextureObject_t opticalflow_map);

		void CompactPotentialValidPixels(cudaStream_t stream = 0);

		// This method, only collect pixel that has non-zero index map value
		// All these pixels are "potentially" matched with depth pixel with appropriate SE3
		void MarkPotentialMatchedPixels(cudaStream_t stream = 0);
		// Considering resampling uniformly
		void MarkPotentialMatchedPixelsResample(
			const GArrayView<float> node_density,
			cudaStream_t stream = 0);
		GArrayView<unsigned> GetSampledIndicator(size_t cam_idx)
		{
			return GArrayView<unsigned>(m_potential_pixel_indicator[cam_idx].ptr(), m_potential_pixel_indicator[cam_idx].size());
		}
		void SyncQueryCompactedPotentialPixelSize(cudaStream_t stream = 0);
		// Unified interface
		void FetchKNNTermSync(cudaStream_t stream);
		void UpdateKnnTermSync(
			const GArrayView<DualQuaternion> node_se3,
			cudaStream_t stream);
		struct ImageTermPixelAndKNN
		{
			GArrayView<ushort4> pixels[d_max_cam];
			GArrayView<unsigned short> surfel_knn_patch[d_max_cam];
			GArrayView<float> knn_patch_spatial_weight[d_max_cam];
			GArrayView<float> knn_patch_connect_weight[d_max_cam];
			GArrayView<DualQuaternion> knn_patch_dq[d_max_cam];
			GArrayView<unsigned short> surfel_knn_patch_all;
			GArrayView<float> knn_patch_spatial_weight_all;
			GArrayView<float> knn_patch_connect_weight_all;
			GArrayView<DualQuaternion> knn_patch_dq_all;
		};
		ImageTermPixelAndKNN GetImageTermPixelAndKNN() const
		{
			ImageTermPixelAndKNN output;
			for (auto cam_idx = 0; cam_idx < m_num_cam; ++cam_idx)
			{
				output.pixels[cam_idx] = m_potential_pixels[cam_idx].View();
				output.surfel_knn_patch[cam_idx] = m_dense_image_knn_patch[cam_idx].View();
				output.knn_patch_spatial_weight[cam_idx] = m_dense_image_knn_patch_spatial_weight[cam_idx].View();
				output.knn_patch_connect_weight[cam_idx] = m_dense_image_knn_patch_connect_weight[cam_idx].View();
				output.knn_patch_dq[cam_idx] = m_dense_image_knn_patch_dq[cam_idx].View();
			}
			output.surfel_knn_patch_all = m_dense_image_knn_patch_all.View();
			output.knn_patch_spatial_weight_all = m_dense_image_knn_patch_spatial_weight_all.View();
			output.knn_patch_connect_weight_all = m_dense_image_knn_patch_connect_weight_all.View();
			output.knn_patch_dq_all = m_dense_image_knn_patch_dq_all.View();
			return output;
		}
		GArrayView<unsigned short> DenseImageTermKNNArray(size_t cam_idx) const { return m_dense_image_knn_patch[cam_idx].View(); }
		GArrayView<unsigned short> DenseImageTermKNNArray() const { return m_dense_image_knn_patch_all.View(); };

		// Sanity check
		void CheckDenseImageTermKNN(size_t cam_idx, const GArrayView<unsigned short> dense_image_knn_gpu);
		// Merge term knn across cameras
		// Fixed
		void MergeTermKNNFixed(cudaStream_t stream);
		// Updated
		void MergeTermKNNUpdated(cudaStream_t stream);

	private:
		// The info from config
		unsigned m_num_cam;
		unsigned m_image_height[d_max_cam];
		unsigned m_image_width[d_max_cam];
		unsigned m_image_height_max;
		unsigned m_image_width_max;
		float m_resample_prob;

		// The info from solver
		struct
		{
			GArrayView2D<KNNAndWeight<d_surfel_knn_size>> knn_patch_map[d_max_cam];
			cudaTextureObject_t index_map[d_max_cam];
			cudaTextureObject_t opticalflow_map[d_max_cam];
		} m_geometry_maps;

		// A fixed size array to indicator the pixel validity
		GArray<unsigned> m_potential_pixel_indicator[d_max_cam];

		// Knn that are fixed
		PrefixSum m_indicator_prefixsum[d_max_cam]; // Sample/Mark for each camera
		GBufferArray<ushort4> m_potential_pixels[d_max_cam];
		GBufferArray<unsigned short> m_dense_image_knn_patch[d_max_cam];
		GBufferArray<float> m_dense_image_knn_patch_spatial_weight[d_max_cam];
		GBufferArray<float> m_dense_image_knn_patch_connect_weight[d_max_cam];

		// Knn that need to be updated
		GBufferArray<DualQuaternion> m_dense_image_knn_patch_dq[d_max_cam];
		curandGenerator_t m_gen; // Used for resampling

		// Counter
		unsigned *m_num_potential_pixel;
		GBufferArray<ushort4> m_potential_pixels_all;
		// KNN-Patch
		GBufferArray<unsigned short> m_dense_image_knn_patch_all;
		GBufferArray<float> m_dense_image_knn_patch_spatial_weight_all;
		GBufferArray<float> m_dense_image_knn_patch_connect_weight_all;
		GBufferArray<DualQuaternion> m_dense_image_knn_patch_dq_all;
	};
} // star