#pragma once

/** \brief
 * Here we summmerize a lot function related with jacobian
 * Provide efficient implementation & unit test for them
 */

// DenseImage related
#include <star/opt/dense_image_jacobian.cuh>
// #include "geometry_jacobian.h"  // Debug
// #include "geometry_jacobian.cuh"
#include <star/opt/dense_image_jacobian_host.h> // For Debug

// NodeGraph Reg related
#include <star/opt/node_graph_reg_jacobian.cuh>
#include <star/opt/node_graph_reg_jacobian_host.h>

// NodeMotion related
#include <star/opt/node_motion_jacobian.cuh>
#include <star/opt/node_motion_jacobian_host.h>