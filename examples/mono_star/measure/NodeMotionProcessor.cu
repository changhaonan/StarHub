#include <mono_star/measure/NodeMotionProcessor.h>
#include <star/geometry/node_graph/node_graph_opticalflow.h>

star::NodeMotionProcessor::NodeMotionProcessor()
{
    auto &config = ConfigParser::Instance();
    m_node_motion_pred.AllocateBuffer(d_max_num_nodes);

    // Camera-setting
    m_downsample_img_cols = config.downsample_img_cols();
    m_downsample_img_rows = config.downsample_img_rows();
    m_cam2world = config.extrinsic()[0];
    m_intrinsic = config.rgb_intrinsic_downsample();

    // Vis
    m_enable_vis = config.enable_vis();
    m_pcd_size = config.pcd_size();
}

star::NodeMotionProcessor::~NodeMotionProcessor()
{
    m_node_motion_pred.ReleaseBuffer();
}

void star::NodeMotionProcessor::ProcessFrame(const SurfelMapTex &surfel_map_this,
                                             const SurfelMapTex &surfel_map_prev,
                                             cudaTextureObject_t opticalflow,
                                             const SurfelGeometry::Geometry4Solver &geometry4solver,
                                             const unsigned num_node,
                                             const unsigned frame_idx,
                                             cudaStream_t stream)
{
    // Run node motion prediction
    computeNodeMotionVisible(
        surfel_map_this,
        surfel_map_prev,
        opticalflow,
        geometry4solver,
        num_node,
        stream);
}

void star::NodeMotionProcessor::ResetNodeMotion(GArraySlice<float4> node_motion_pred, cudaStream_t stream)
{
    cudaSafeCall(cudaMemsetAsync(node_motion_pred.Ptr(), 0, sizeof(float4) * node_motion_pred.Size(), stream));
}

void star::NodeMotionProcessor::computeNodeMotionVisible(
    const SurfelMapTex &surfel_map_this,
    const SurfelMapTex &surfel_map_prev,
    cudaTextureObject_t opticalflow,
    const SurfelGeometry::Geometry4Solver &geometry4solver,
    const unsigned num_node,
    cudaStream_t stream)
{
    // Reset node motion
    m_node_motion_pred.ResizeArrayOrException(num_node);
    ResetNodeMotion(m_node_motion_pred.Slice(), stream);

    AccumlateNodeMotionFromOpticalFlow(
        surfel_map_prev.vertex_confid,
        surfel_map_this.vertex_confid,
        surfel_map_prev.rgbd,
        surfel_map_this.rgbd,
        opticalflow,
        surfel_map_prev.index,
        geometry4solver.surfel_knn,
        geometry4solver.surfel_knn_spatial_weight,
        m_node_motion_pred.Slice(),
        m_cam2world,
        m_intrinsic,
        stream);

    AverageNodeMotion(m_node_motion_pred.Slice(), stream);
    // Sync before exit
    cudaSafeCall(cudaStreamSynchronize(stream));
}