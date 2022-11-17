#pragma once
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>
#include <star/common/common_utils.h>
#include <star/common/ArrayView.h>
#include <star/common/logging.h>
#include <star/math/DualQuaternion.hpp>
#include <star/math/DualQuaternion.hpp>
#include <star/common/constants.h>
#include <star/geometry/constants.h>
#include <star/opt/constants.h>
#include <ostream>

namespace star
{
    /* Input for solver, Measurement, 2D, image-level
     */
    struct Measure4Solver
    {
        cudaTextureObject_t vertex_confid_map[d_max_cam];
        cudaTextureObject_t normal_radius_map[d_max_cam];
        cudaTextureObject_t index_map[d_max_cam];
        unsigned num_cam;
    };

    /* Input for solver, Render, 2D, image-level
     */
    struct Render4Solver
    {
        cudaTextureObject_t reference_vertex_map[d_max_cam];
        cudaTextureObject_t reference_normal_map[d_max_cam];
        cudaTextureObject_t index_map[d_max_cam];
        unsigned num_cam;
    };

    // Solver map
    struct SolverMaps
    {
        using Ptr = std::shared_ptr<SolverMaps>;

        cudaTextureObject_t reference_vertex_map[d_max_cam] = {0};
        cudaTextureObject_t reference_normal_map[d_max_cam] = {0};
        cudaTextureObject_t index_map[d_max_cam] = {0};
        cudaTextureObject_t normalized_rgbd_map[d_max_cam] = {0};
    };

    /* Input for solver, Geometry, 1D, surfel-level
     */
    struct Geometry4Solver
    {
        GArrayView<ushortX<d_surfel_knn_size>> surfel_knn;
        GArrayView<floatX<d_surfel_knn_size>> surfel_knn_spatial_weight;
        GArrayView<floatX<d_surfel_knn_size>> surfel_knn_connect_weight;
        unsigned num_vertex;
    };

    /* Input for solver, NodeGraph, 1D, node-level
     */
    struct NodeGraph4Solver
    {
        // Used for reg term
        GArrayView<float4> reference_node_coords;
        GArrayView<ushort3> node_graph;
        GArrayView<floatX<d_node_knn_size>> node_knn_connect_weight;
        // Used for node motion term
        GArrayView<ushortX<d_surfel_knn_size>> nodel_knn;
        GArrayView<floatX<d_surfel_knn_size>> node_knn_spatial_weight;
        unsigned num_node;
        float node_radius_square;
    };

    /* Input for solver, NodeFlow, 1D, node-level
     */
    struct NodeFlow4Solver
    {
        GArrayView<float4> node_motion_pred;
        unsigned num_node;
    };

    /* Input for solver, OpticalFlow, 2D, image-level
     */
    struct OpticalFlow4Solver
    {
        cudaTextureObject_t opticalflow_map[d_max_cam];
        unsigned num_cam;
    };

    /**
     * \brief The gradient of some scalar cost
     *        towards the twist parameters of node
     *        SE(3) INCREMENT, while the node SE(3)
     *        itself is parameterized by DualQuaternion
     */
    struct TwistGradientOfScalarCost
    {
        float3 rotation;
        float3 translation;

        // Default zero initialization
        __host__ __device__ TwistGradientOfScalarCost()
        {
            rotation = make_float3(0.f, 0.f, 0.f);
            translation = make_float3(0.f, 0.f, 0.f);
        }

        // Constant times operator
        __host__ TwistGradientOfScalarCost operator*(const float &value) const
        {
            TwistGradientOfScalarCost timed_twist = *this;
            timed_twist.rotation *= value;
            timed_twist.translation *= value;
            return timed_twist;
        }

        // Dot with a size 6 array
        __host__ __device__ float dot(const float x[6]) const
        {
            const float rot_dot = x[0] * rotation.x + x[1] * rotation.y + x[2] * rotation.z;
            const float trans_dot = x[3] * translation.x + x[4] * translation.y + x[5] * translation.z;
            return rot_dot + trans_dot;
        }

        // Dot with a texture memory
        __device__ __forceinline__ float DotLinearTexture(cudaTextureObject_t x, const unsigned x_offset) const
        {
            const float *jacobian = (const float *)(this);
            float dot_value = 0.0f;
            for (auto i = 0; i < 6; i++)
            {
                dot_value += jacobian[i] * fetch1DLinear<float>(x, x_offset + i);
            }
            return dot_value;
        }
        __device__ __forceinline__ float DotArrayTexture(cudaTextureObject_t x, const unsigned x_offset) const
        {
            const float *jacobian = (const float *)(this);
            float dot_value = 0.0f;
            for (auto i = 0; i < 6; i++)
            {
                dot_value += jacobian[i] * fetch1DArray<float>(x, x_offset + i);
            }
            return dot_value;
        }
    };

    /**
     * \brief The gradient of some scalar cost
     *        towards the twist parameters of node
     *        SE(3) INCREMENT and weight change, while the node SE(3)
     *        itself is parameterized by DualQuaternion
     */
    struct TwistWeightGradientOfScalarCost
    {
        float3 rotation;
        float3 translation;
        float weight[d_node_knn_size];

        // Constant times operator
        __host__ __device__ __forceinline__
            TwistWeightGradientOfScalarCost
            operator*(const float &value) const
        {
            TwistWeightGradientOfScalarCost timed_gradient = *this;
            timed_gradient.rotation *= value;
            timed_gradient.translation *= value;
#pragma unroll
            for (auto i = 0; i < d_node_knn_size; ++i)
            {
                timed_gradient.weight[i] *= value;
            }
            return timed_gradient;
        }
    };

    /* The gradient with multiple scalar cost
     */
    template <int NumChannels = 3>
    struct TwistGradientChannel
    {
        TwistGradientOfScalarCost gradient[NumChannels];
    };

    // Binding
    using GradientOfScalarCost = TwistGradientOfScalarCost;
    using GradientOfRegCost = TwistGradientOfScalarCost;

    /* Lie algebra representation
     */
    struct Lie
    {
        float lie[6];
    };

    /**
     * \brief The Term2Jacobian structs, as its name suggested,
     *        provide enough information to compute the gradient
     *        of the cost from a given term index w.r.t all the
     *        nodes that this term is involved.
     *        Note that each term may have ONE or MORE scalar
     *        costs. In case of multiple scalar costs, the jacobian
     *        CAN NOT be combined.
     *		  The jacobian consists of one part: 1). Twist: gradient to
     *        lie algebra.
     *        The term2jacobian should implemented on device, but
     *        may provide host implementation for debug checking.
     */
    struct ScalarCostTerm2Jacobian
    {
        GArrayView<unsigned short> knn_patch_array;
        GArrayView<float> knn_patch_spatial_weight_array;
        GArrayView<float> knn_patch_connect_weight_array;
        GArrayView<DualQuaternion> knn_patch_dq_array;
        GArrayView<float> residual_array;
        // Gradient
        GArrayView<GradientOfScalarCost> gradient_array;

        // Simple sanity check
        __host__ __forceinline__ void check_size() const
        {
            STAR_CHECK_EQ(knn_patch_array.Size(), knn_patch_spatial_weight_array.Size());
            STAR_CHECK_EQ(knn_patch_array.Size(), knn_patch_connect_weight_array.Size());
            STAR_CHECK_EQ(knn_patch_array.Size(), knn_patch_dq_array.Size());
            STAR_CHECK_EQ(knn_patch_array.Size(), residual_array.Size() * d_node_knn_size);
            STAR_CHECK_EQ(knn_patch_array.Size(), gradient_array.Size() * d_node_knn_size);
        }
    };

    template <int num_channel>
    struct VectorCostTerm2Jacobian
    {
        GArrayView<unsigned short> knn_patch_array;
        GArrayView<float> knn_patch_spatial_weight_array;
        GArrayView<float> knn_patch_connect_weight_array;
        GArrayView<DualQuaternion> knn_patch_dq_array;
        GArrayView<floatX<num_channel>> residual_array;
        // Gradient
        GArrayView<TwistGradientChannel<num_channel>> gradient_array;

        // Simple sanity check
        __host__ __forceinline__ void check_size() const
        {
            STAR_CHECK_EQ(knn_patch_array.Size(), knn_patch_spatial_weight_array.Size());
            STAR_CHECK_EQ(knn_patch_array.Size(), knn_patch_connect_weight_array.Size());
            STAR_CHECK_EQ(knn_patch_array.Size(), knn_patch_dq_array.Size());
            STAR_CHECK_EQ(knn_patch_array.Size(), residual_array.Size() * d_node_knn_size);
            STAR_CHECK_EQ(knn_patch_array.Size(), gradient_array.Size() * d_node_knn_size);
        }
    };

    // These are all scalar cost term types
    using GradientOfDenseImage = TwistGradientChannel<d_dense_image_residual_dim>;
    using DenseImageTerm2Jacobian = VectorCostTerm2Jacobian<d_dense_image_residual_dim>;
    using FeatureTerm2Jacobian = ScalarCostTerm2Jacobian;
    /**
     * \brief It seems cheaper to compute the jacobian online
     *        for Reg term.
     */
    struct NodeGraphRegTerm2Jacobian
    {
        GArrayView<ushort3> node_graph; // (i, j, k). j is the kth neighbor of i.
        GArrayView<float3> Ti_xj;
        GArrayView<float3> Tj_xj;
        GArrayView<unsigned char> validity_indicator;
        GArrayView<float> connect_weight;
    };

    struct NodeTranslationTerm2Jacobian
    {
        GArrayView<float3> T_translation;
        GArrayView<float4> node_motion_pred;
    };

    /* The collective term2jacobian maps
     */
    struct Term2JacobianMaps
    {
        DenseImageTerm2Jacobian dense_image_term;
        NodeGraphRegTerm2Jacobian node_graph_reg_term;
        NodeTranslationTerm2Jacobian node_translation_term;
        FeatureTerm2Jacobian feature_term;
    };

    /* The node-wise error and weight, compute only use dense depth information
     */
    struct NodeAlignmentError
    {
        GArrayView<float> node_accumulated_error;
        const float *node_accumulate_weight;

        // The statistic method
        __host__ void errorStatistics(std::ostream &output = std::cout) const;
    };

}