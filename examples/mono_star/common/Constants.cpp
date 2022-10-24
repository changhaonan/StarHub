#include "Constants.h"

const float star::Constants::kFilterSigma_S = 4.5f;
const float star::Constants::kFilterSigma_R = 30.0f;

// The maximum number of sparse feature terms
const unsigned star::Constants::kMaxMatchedSparseFeature = 32; // Reserve for future version

// The scale of fusion map
const int star::Constants::kFusionMapScale = d_fusion_map_scale;
const int star::Constants::kFilterMapScale = d_fusion_map_scale; // Currently set to the same; not used yet

// The maximum number of surfels
const unsigned star::Constants::kMaxNumSurfels = 800000; // Change according to need

// The maximum number of nodes
const unsigned star::Constants::kMaxNumNodes = d_max_num_nodes;
const unsigned star::Constants::kMaxNumNodePairs = star::Constants::kMaxNumNodes * 60; // Each node can be connected with 60
const unsigned star::Constants::kMaxNumSurfelCandidates = 400000;

// The number of node graph neigbours
const unsigned star::Constants::kNumNodeGraphNeigbours = d_node_knn_size;

// The recent time threshold for rendering solver maps
const int star::Constants::kRenderingRecentTimeThreshold = 3;

// The confidence threshold for stable surfel
const int star::Constants::kStableSurfelConfidenceThreshold = 1;

// Use elastic penalty or not
const bool star::Constants::kUseElasticPenalty = true;
const int star::Constants::kNumGlobalSolverItarations = 3;
// const int star::Constants::kNumGaussNewtonIterations = 6;
const int star::Constants::kNumGaussNewtonIterations = 10;