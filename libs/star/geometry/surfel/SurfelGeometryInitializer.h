#pragma once
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>
#include <star/common/surfel_types.h>
#include <star/geometry/constants.h>
#include <star/geometry/surfel/SurfelGeometry.h>
#include <memory>

namespace star
{
	class SurfelGeometryInitializer
	{
	public:
		// The processer interface
		static void InitFromObservationSerial(
			SurfelGeometry &geometry,
			const GArrayView<DepthSurfel> &surfel_array,
			cudaStream_t stream = 0);
		static void InitFromMultiObservationSerial(
			SurfelGeometry &geometry,
			const unsigned num_cam,
			const GArrayView<DepthSurfel> *surfel_arrays,
			const Eigen::Matrix4f *cam2world,
			cudaStream_t stream = 0);
		static void InitFromMultiObservationSerial(
			LiveSurfelGeometry &geometry,
			const unsigned num_cam,
			const GArrayView<DepthSurfel> *surfel_arrays,
			const Eigen::Matrix4f *cam2world,
			cudaStream_t stream = 0);
		static void InitFromDataGeometry(
			SurfelGeometry &geometry,
			SurfelGeometry &data_geometry,
			const bool use_semantic,
			cudaStream_t stream = 0);

		// The members from other classes
		using GeometryAttributes = SurfelGeometry::GeometryAttributes;
		using LiveGeometryAttributes = LiveSurfelGeometry::GeometryAttributes;

		/* Collect the compacted valid surfel array into geometry
		 */
	private:
		static void initSurfelGeometry(
			GeometryAttributes geometry,
			const GArrayView<DepthSurfel> &surfel_array,
			cudaStream_t stream = 0);
		static void initSurfelGeometry( // Multi-observation initializer
			GeometryAttributes geometry,
			const unsigned num_cam,
			const GArrayView<DepthSurfel> *surfel_arrays,
			const Eigen::Matrix4f *cam2world,
			cudaStream_t stream = 0);
		static void initSurfelGeometry( // Multi-observation initializer for live geometry
			LiveGeometryAttributes geometry,
			const unsigned num_cam,
			const GArrayView<DepthSurfel> *surfel_arrays,
			const Eigen::Matrix4f *cam2world,
			cudaStream_t stream = 0);
	};
}