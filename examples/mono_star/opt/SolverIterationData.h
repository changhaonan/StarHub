#pragma once
#include <star/common/GBufferArray.h>
#include <star/math/DualQuaternion.hpp>
#include <star/geometry/node_graph/NodeGraph.h>
#include <mono_star/common/ConfigParser.h>
#include <mono_star/common/Constants.h>
#include <mono_star/opt/PenaltyConstants.h>

namespace star
{
    /*
     * Both se3 & connect weight will be updated during optimization
     */
    class SolverIterationData
    {
    private:
        // The state to keep track current input/output
        enum class IterationInputFrom
        {
            WarpFieldInit,
            Buffer_0,
            Buffer_1
        };

        // The input from warp field
        GBufferArray<DualQuaternion> m_node_se3_init;

        // The double buffer are maintained in this class
        GBufferArray<DualQuaternion> m_node_se3_0;
        GBufferArray<DualQuaternion> m_node_se3_1;

        IterationInputFrom m_updated_warpfield;
        unsigned m_newton_iters;
        void updateIterationFlags();

        // Only need to keep one joint (twist, connection_weight) buffer
        GBufferArray<float> m_warpfield_update;

        // The constants for different terms
        PenaltyConstants m_penalty_constants;
        void setElasticPenaltyValue(int newton_iter, PenaltyConstants &constants);

        // Switch for global interation
        bool m_is_global_iteration;

        // Allocate and release buffers
        void allocateBuffer();
        void releaseBuffer();

    public:
        explicit SolverIterationData();
        ~SolverIterationData();
        STAR_NO_COPY_ASSIGN_MOVE(SolverIterationData);

        // The process interface
        void SetWarpFieldInitialValue(const unsigned num_nodes);
        bool IsInitialIteration() const { return m_updated_warpfield == IterationInputFrom::WarpFieldInit; }

        // General fetch api
        GArrayView<DualQuaternion> CurrentNodeSE3Input() const;
        GArrayView<DualQuaternion> NodeSE3Init() const { return m_node_se3_init.View(); }
        // Update Buffer
        GArraySlice<float> CurrentWarpFieldUpdateBuffer();

        // The constants for current iteration
        PenaltyConstants CurrentPenaltyConstants() const { return m_penalty_constants; }
        bool ComputeJtJLazyEvaluation() const { return m_newton_iters >= Constants::kNumGlobalSolverItarations; };
        bool IsGlobalIteration() const { return m_is_global_iteration; }

        // External accessed sanity check method
        void SanityCheck() const;
        unsigned NumNodes() const { return m_node_se3_init.ArraySize(); }

        // Required cuda access
        void ApplyWarpFieldUpdate(cudaStream_t stream = 0, float se3_step = 1.0f);

        // Initialization method
        void InitializedAsIdentity(const unsigned num_nodes, cudaStream_t stream = 0);
    };

}