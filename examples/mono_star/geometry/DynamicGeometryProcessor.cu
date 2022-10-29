#include <mono_star/common/ConfigParser.h>
#include "DynamicGeometryProcessor.h"

star::DynamicGeometryProcessor::DynamicGeometryProcessor()
{
    auto &config = ConfigParser::Instance();

    m_data_geometry = std::make_shared<star::SurfelGeometry>();
    m_model_geometry[0] = std::make_shared<star::SurfelGeometry>();
    m_model_geometry[1] = std::make_shared<star::SurfelGeometry>();

    // Render
    m_renderer = std::make_shared<star::Renderer>(
        config.num_cam(),
        config.downsample_img_cols(),
        config.downsample_img_rows(),
        config.rgb_intrinsic_downsample(),
        config.max_rendering_depth()
    );
    
    m_renderer->MapDataSurfelGeometryToCuda(0, *m_data_geometry);
    m_renderer->MapModelSurfelGeometryToCuda(0, *m_model_geometry[0]);
	m_renderer->MapModelSurfelGeometryToCuda(1, *m_model_geometry[1]);
}

star::DynamicGeometryProcessor::~DynamicGeometryProcessor()
{
    m_renderer->UnmapDataSurfelGeometryFromCuda(0);
	m_renderer->UnmapModelSurfelGeometryFromCuda(0);
	m_renderer->UnmapModelSurfelGeometryFromCuda(1);
}

void star::DynamicGeometryProcessor::Process(
    StarStageBuffer &star_stage_buffer_this,
    const StarStageBuffer &star_stage_buffer_prev,
    cudaStream_t stream,
    const unsigned frame_idx)
{
}

void star::DynamicGeometryProcessor::processFrame(
    const unsigned frame_idx,
    cudaStream_t stream)
{
}

void star::DynamicGeometryProcessor::initGeometry(const GeometryMap &geometry_map)
{
    // Init data geometry

}