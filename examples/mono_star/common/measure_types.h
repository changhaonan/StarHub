#pragma once
#include <star/common/common_types.h>
#include <star/common/common_texture_utils.h>
#include <star/geometry/geometry_types.h>
#include "ConfigParser.h"
#include "frame_buffer.h"

namespace star {

    struct SingleMeasureMap {
        cudaTextureObject_t vertex_map;
        cudaTextureObject_t normal_map;
        cudaTextureObject_t color_time_map;
        cudaTextureObject_t rgbd_map;           // Used for optical-flow
        cudaTextureObject_t raw_rgb_map;        // Used for segmentation  #TODO: merge them together
        cudaTextureObject_t depth4removal_map;   // Used for geometry removal
        cudaTextureObject_t index_map;
        unsigned valid_pixel_num;
    };


    struct SingleMeasureBuffer {
        CudaTextureSurface vertex_confid;
        CudaTextureSurface normal_radius;
        CudaTextureSurface color_time;
        CudaTextureSurface rgbd;        // (R,G,B,D_inv) scaled to -1~1, D_inv is 0~1
        CudaTextureSurface raw_rgb;     // Raw RGB without any resizing, only scaled to 0~1
        CudaTextureSurface depth4removal;
        CudaTextureSurface index;
        unsigned valid_pixel_num;       // Used for flat
        unsigned index_offset;

        void create(
            const unsigned raw_img_rows, const unsigned raw_img_cols,
            const unsigned downsample_img_rows, const unsigned downsample_img_cols) {
            createFloat4TextureSurface(downsample_img_rows, downsample_img_cols, vertex_confid);
            createFloat4TextureSurface(downsample_img_rows, downsample_img_cols, normal_radius);
            createFloat4TextureSurface(downsample_img_rows, downsample_img_cols, color_time);
            createFloat4TextureSurface(downsample_img_rows, downsample_img_cols, rgbd);
            createFloat4TextureSurface(raw_img_rows, raw_img_cols, raw_rgb);
            createFloat1TextureSurface(downsample_img_rows, downsample_img_cols, depth4removal);
            createIndexTextureSurface(downsample_img_rows, downsample_img_cols, index);
        }
        void release() {
            releaseTextureCollect(vertex_confid);
            releaseTextureCollect(normal_radius);
            releaseTextureCollect(color_time);
            releaseTextureCollect(rgbd);
            releaseTextureCollect(raw_rgb);
            releaseTextureCollect(depth4removal);
            releaseTextureCollect(index);
        }

        operator SingleMeasureMap() const {  // Auto-transfer
            SingleMeasureMap measure_map;
            measure_map.vertex_map = vertex_confid.texture;
            measure_map.normal_map = normal_radius.texture;
            measure_map.color_time_map = color_time.texture;
            measure_map.rgbd_map = rgbd.texture;
            measure_map.raw_rgb_map = raw_rgb.texture;
            measure_map.depth4removal_map = depth4removal.texture;
            measure_map.index_map = index.texture;
            measure_map.valid_pixel_num = valid_pixel_num;
            return measure_map;
        }
    };


	struct MeasureBuffer : FrameBuffer {
        MeasureBuffer() { create(); }
        ~MeasureBuffer() { release(); }
        void create() override {
            auto& config = ConfigParser::Instance();
            for (auto cam_idx = 0; cam_idx < config.num_cam(); ++cam_idx) {
                buffer[cam_idx].create(
                    config.raw_img_rows(cam_idx),
                    config.raw_img_cols(cam_idx),
                    config.downsample_img_rows(cam_idx), 
                    config.downsample_img_cols(cam_idx));
            }
        }
        void release() override {
            auto& config = ConfigParser::Instance();
            for (auto cam_idx = 0; cam_idx < config.num_cam(); ++cam_idx) {
                buffer[cam_idx].release();
            }
        }

        // Fetch-API
        SingleMeasureMap at(const unsigned cam_idx) const {
            return (SingleMeasureMap)buffer[cam_idx];
        }
        SingleMeasureBuffer buffer[d_max_cam];
        unsigned num_valid_surfel;

		void log(const unsigned buffer_idx) override {
            std::string text = stringFormat(" Measure - frame %d - buffer: %d ", frame_idx, buffer_idx);
            std::string aligned_text = stringAlign2Center(text, logging_length, "=");
            std::cout << aligned_text << std::endl;
		};

        // Access for other module
        Measure4Solver GenerateMeasure4Solver(const unsigned num_cam) const {
            Measure4Solver measure4solver{};
            for (auto cam_idx = 0; cam_idx < num_cam; ++cam_idx) {
                measure4solver.vertex_confid_map[cam_idx]
                    = buffer[cam_idx].vertex_confid.texture;
                measure4solver.normal_radius_map[cam_idx]
                    = buffer[cam_idx].normal_radius.texture;
                measure4solver.index_map[cam_idx]
                    = buffer[cam_idx].index.texture;
                measure4solver.num_cam = num_cam;
            }   
            return measure4solver;
        }

        Measure4Fusion GenerateMeasure4Fusion(const unsigned num_cam) const {
            Measure4Fusion measure4fusion{};
            for (auto cam_idx = 0; cam_idx < num_cam; ++cam_idx) {
                measure4fusion.vertex_confid_map[cam_idx]
                    = buffer[cam_idx].vertex_confid.texture;
                measure4fusion.normal_radius_map[cam_idx]
                    = buffer[cam_idx].normal_radius.texture;
                measure4fusion.color_time_map[cam_idx]
                    = buffer[cam_idx].color_time.texture;
                measure4fusion.index_map[cam_idx]
                    = buffer[cam_idx].index.texture;
            }
            measure4fusion.num_valid_surfel = num_valid_surfel;
            return measure4fusion;
        }

        Measure4GeometryRemoval GenerateMeasure4GeometryRemoval(const unsigned num_cam) const {
            Measure4GeometryRemoval measure4geometry_removal{};
            for (auto cam_idx = 0; cam_idx < num_cam; ++cam_idx) {
                measure4geometry_removal.depth4removal_map[cam_idx]
                    = buffer[cam_idx].depth4removal.texture;
            }
            return measure4geometry_removal;
        }
	};

}