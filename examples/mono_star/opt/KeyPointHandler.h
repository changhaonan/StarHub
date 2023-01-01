#pragma once
#include <star/common/macro_utils.h>
#include <star/common/ArrayView.h>
#include <star/common/GBufferArray.h>
#include <star/math/DualQuaternion.hpp>
#include <star/opt/solver_types.h>

namespace star
{
    class KeyPointHandler
    {
    private:
        // The input data from keypoint
        GArrayView<float4> m_kp_vertex_confid;
        GArrayView<float4> m_kp_normal_radius;
        // The input data from detected keypoint
        GArrayView<float4> m_d_kp_vertex_confid;
        GArrayView<float4> m_d_kp_normal_radius;
        // Match-related
        GArrayView<int2> m_kp_match; // match = (id_measure, id_model)
        unsigned m_num_match;

        // KNN structure
        GArrayView<ushortX<d_surfel_knn_size>> m_kp_knn;
        GArrayView<floatX<d_surfel_knn_size>> m_kp_knn_spatial_weight;
        GArrayView<floatX<d_surfel_knn_size>> m_kp_knn_connect_weight;
        // The info from solver
        GArrayView<DualQuaternion> m_node_se3;
        // The output
        GBufferArray<float> m_term_residual;
        GBufferArray<TwistGradientOfScalarCost> m_term_gradient;
        // Intermidate (update each time)
        GBufferArray<unsigned short> m_knn_patch_array;
        GBufferArray<float> m_knn_patch_spatial_weight_array;
        GBufferArray<float> m_knn_patch_connect_weight_array;
        GBufferArray<DualQuaternion> m_knn_patch_dq_array;
        /* Compute the twist jacobian
         */
    public:
        using Ptr = std::shared_ptr<KeyPointHandler>;
        KeyPointHandler();
        ~KeyPointHandler();
        STAR_NO_COPY_ASSIGN_MOVE(KeyPointHandler);

        // Explicit allocate
        void AllocateBuffer();
        void ReleaseBuffer();
        void SetInputs(
            const KeyPoint4Solver &keypoint4solver);
        void UpdateInputs(
            const GArrayView<DualQuaternion> &node_se3,
            cudaStream_t stream);
        void InitKNNSync(cudaStream_t stream);
        void ResizeNumMatch(const unsigned num_match);
        unsigned NumValidMatch() const { return m_num_match; }
        // Output
        FeatureTerm2Jacobian Term2JacobianMap() const;
        GArrayView<unsigned short> KNNPatchArray() const { return m_knn_patch_array.View(); };
        // Debug method
        void DebugResidual();
        // Jacobian method
        void BuildTerm2Jacobian(cudaStream_t stream);
        void ComputerJacobianTermsFixedIndex(cudaStream_t stream);
    };
}