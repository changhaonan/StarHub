constexpr unsigned d_transform_dim = 6;
//constexpr unsigned d_node_variable_dim = (d_transform_dim + d_node_knn_size);
constexpr unsigned d_node_variable_dim = d_transform_dim;  // Only update for transformation now
constexpr unsigned d_node_variable_dim_square = d_node_variable_dim * d_node_variable_dim;
constexpr unsigned d_dense_image_residual_dim = 3;  // 1 (picp) + 2 (opticalflow)