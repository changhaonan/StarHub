#pragma once
#include <star/common/common_types.h>
#include <star/io/VolumeDeformFileFetch.h>
#include <star/geometry/geometry_map/SurfelMap.h>
#include <star/geometry/geometry_map/SurfelMapInitializer.h>
#include <super_recon/common/ConfigParser.h>
// Viewer
#include <easy3d_viewer/context.hpp>

namespace star
{

	/* \brief Read measure from offline file
	 */
	class MeasureProcessorOffline
	{
	public:
		using Ptr = std::shared_ptr<MeasureProcessorOffline>;
		STAR_NO_COPY_ASSIGN_MOVE(MeasureProcessorOffline);
		MeasureProcessorOffline();
		~MeasureProcessorOffline();
		void ProcessFrame(
			const unsigned frame_idx,
			cudaStream_t stream);
		void saveContext(
			const unsigned frame_idx,
			cudaStream_t stream);

		// Fetch-API
		SurfelMap::Ptr GetSurfelMap() {
			return m_surfel_map;
		}
		SurfelMap::Ptr GetSurfelMapReadOnly() const {
			return m_surfel_map;
		}
		SurfelMapTex GetSurfelMapTex() const { return m_surfel_map->Texture(); };
	private:
		void drawOrigin();

		unsigned m_start_frame_idx;
		unsigned m_step_frame;
		VolumeDeformFileFetch::Ptr m_fetcher;

		// Buffer data
		cv::Mat m_raw_depth_img;
		cv::Mat m_raw_color_img;
		void* m_raw_depth_img_buff;
		void* m_raw_color_img_buff;

		GArray<uchar3> m_g_raw_color_img;
		GArray<unsigned short> m_g_raw_depth_img;
		SurfelMap::Ptr m_surfel_map;
		SurfelMap::Ptr m_surfel_map_prev;
		SurfelMapInitializer::Ptr m_surfel_map_initializer;

		// Camera-related
		float m_downsample_scale;
		Eigen::Matrix4f m_cam2world;  // Camera position

		// Visualize-related
		bool m_enable_vis;
		float m_pcd_size;
	};

}