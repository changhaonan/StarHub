#pragma once
#include <string>
#include "global_configs.h"

namespace star
{

	/**
	 * \brief The struct to maintained the
	 *        host accessed constants.
	 */
	struct Constants
	{
		// The sigma value used in bilateral filtering
		const static float kFilterSigma_S;
		const static float kFilterSigma_R;

		// The size required by the renderer
		const static int kFusionMapScale;
		const static int kFilterMapScale;

		// The maximum number of surfels
		const static unsigned kMaxNumSurfels;

		// The maximum number of nodes and valid pairs
		const static unsigned kMaxNumNodes;
		const static unsigned kMaxNumNodePairs;
		const static unsigned kMaxNumSurfelCandidates;

		// The number of node graph neigbours
		const static unsigned kNumNodeGraphNeigbours;

		// The recent time threshold for rendering solver maps
		const static int kRenderingRecentTimeThreshold;

		// The confidence threshold for stable surfel
		const static int kStableSurfelConfidenceThreshold;

		// The maximum number of sparse feature terms
		const static unsigned kMaxMatchedSparseFeature;

		// Use elastic penalty and the number of them
		const static bool kUseElasticPenalty;
		const static int kNumGlobalSolverItarations;
		const static int kNumGaussNewtonIterations;
	};
}