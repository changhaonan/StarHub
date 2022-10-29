#pragma once
#include <star/common/common_texture_utils.h>
#include <star/geometry/surfel/SurfelGeometry.h>
#include <star/geometry/node_graph/NodeGraph.h>
#include "frame_buffer.h"

namespace star
{

	// Debug type, to be viewed
	struct Geometry4Monitor
	{
		GArrayView<float4> appended_vertex_confid;
		GArrayView<float4> removed_vertex_confid;
		GArrayView<float4> unsupported_candidate;
	};

	struct Render4Solver
    {
        cudaTextureObject_t reference_vertex_map[d_max_cam];
        cudaTextureObject_t reference_normal_map[d_max_cam];
        cudaTextureObject_t index_map[d_max_cam];
        unsigned num_cam;
    };

	struct DynamicGeometryBuffer : FrameBuffer
	{
		DynamicGeometryBuffer()
		{
			create();
			surfel_geometry = nullptr;
			node_graph = nullptr;
		}
		~DynamicGeometryBuffer() { release(); }
		void create() override
		{
			auto &config = ConfigParser::Instance();
			num_cam = config.num_cam();
#ifdef DYNAMIC_GEOMETRY_DEBUG
			for (auto cam_idx = 0; cam_idx < num_cam; ++cam_idx)
			{
				createFloat4TextureSurface(
					config.downsample_img_rows(cam_idx),
					config.downsample_img_cols(cam_idx),
					reference_vertex_confid_map[cam_idx]);
			}
			reference_vertex_confid_array.AllocateBuffer(Constants::kMaxNumSurfels);
			reference_node_coordinate.AllocateBuffer(Constants::kMaxNumNodes);
			appended_vertex_confid.AllocateBuffer(Constants::kMaxNumSurfelCandidates);
			removed_vertex_confid.AllocateBuffer(Constants::kMaxNumNodes);
			unsupported_candidate.AllocateBuffer(Constants::kMaxNumSurfelCandidates);
#endif
		}
		void release() override
		{
#ifdef DYNAMIC_GEOMETRY_DEBUG
			for (auto cam_idx = 0; cam_idx < num_cam; ++cam_idx)
			{
				releaseTextureCollect(reference_vertex_confid_map[cam_idx]);
			}
			reference_vertex_confid_array.ReleaseBuffer();
			reference_node_coordinate.ReleaseBuffer();
			appended_vertex_confid.ReleaseBuffer();
			removed_vertex_confid.ReleaseBuffer();
			unsupported_candidate.ReleaseBuffer();
#endif
		}

		Render4Solver GenerateRender4Solver(const unsigned num_cam) const
		{
			Render4Solver render4solver;
			// render4solver.num_cam = num_cam;
			// for (auto cam_idx = 0; cam_idx < num_cam; ++cam_idx)
			// {
			// 	render4solver.reference_vertex_map[cam_idx] = solver_maps.reference_vertex_map[cam_idx];
			// 	render4solver.reference_normal_map[cam_idx] = solver_maps.reference_normal_map[cam_idx];
			// 	render4solver.index_map[cam_idx] = solver_maps.index_map[cam_idx];
			// }
			return render4solver;
		}

		// Shared
		SurfelGeometry::Ptr surfel_geometry;   // Shared with the processor
		NodeGraph::Ptr node_graph;			   // Shared with the processor

		// Shared for debug
		GArrayView<float4> append_candidate;

		// Owned (For debug)
		unsigned num_cam;
#ifdef DYNAMIC_GEOMETRY_DEBUG
		CudaTextureSurface reference_vertex_confid_map[d_max_cam]; // Reference vertex in this epoch [In world coordinate]
		GBufferArray<float4> reference_vertex_confid_array;
		GBufferArray<float4> reference_node_coordinate;
		// For monitor
		GBufferArray<float4> appended_vertex_confid;
		GBufferArray<float4> removed_vertex_confid;
		GBufferArray<float4> unsupported_candidate;

		Geometry4Monitor GenerateGeometry4Monitor() const
		{
			Geometry4Monitor geometry4monitor{};
			geometry4monitor.appended_vertex_confid = appended_vertex_confid.View();
			geometry4monitor.removed_vertex_confid = removed_vertex_confid.View();
			geometry4monitor.unsupported_candidate = unsupported_candidate.View();
			return geometry4monitor;
		}
#endif // DYNAMIC_GEOMETRY_DEBUG

		void log(const unsigned buffer_idx) override
		{
			std::string text = stringFormat(" Dynamic Geometry - frame %d - buffer: %d ", frame_idx, buffer_idx);
			std::string aligned_text = stringAlign2Center(text, logging_length, "=");
			std::cout << aligned_text << std::endl;
		};
	};
}