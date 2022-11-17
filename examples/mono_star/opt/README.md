## Structure of warpsolver

### IO

The input of warpsolver is **M<sup>t</sup>** (Measurement), **N<sup>t-1</sup>** (Node graph), **R<sup>t-1</sup>** (Render), **G<sup>t-1</sup>** (Geometry).

The details of input:

- M<sup>t</sup> = { Vertex<sup>t</sup>, Normal<sup>t</sup>, Index<sup>t</sup> }. Data are defined on measurement 2D coordinates.

- G<sup>t-1</sup> = { Knn<sup>t-1</sup>, KnnW<sup>t-1</sup> }. Data are defined on geometry 1D coordinate.

- R<sup>t-1</sup> = { Vertex<sup>t-1</sup>, Normal<sup>t-1</sup>, Index<sup>t-1</sup>, OpticalFlow<sup>t-1</sup>, } Data are defined on geometry 2D coordinate.

- N<sup>t-1</sup> = { SE3<sup>t-1</sup>, Ref<sup>t-1</sup>, NodeGraph<sup>t-1</sup>, NodeTranslation<sup>t-1</sup>, NodeKNNConnect<sup>t-1</sup> }