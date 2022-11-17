#pragma once
#include <star/common/macro_utils.h>
#include <star/common/sanity_check.h>
#include <star/pcg_solver/ApplySpMVBase.h>

template <int BlockDim>
void star::ApplySpMVBase<BlockDim>::CompareApplySpMV(typename ApplySpMVBase<BlockDim>::Ptr applier_0, typename ApplySpMVBase<BlockDim>::Ptr applier_1)
{
	STAR_CHECK(applier_0->MatrixSize() == applier_1->MatrixSize());

	// Prepare the data
	std::vector<float> x_h;
	x_h.resize(applier_0->MatrixSize());
	fillRandomVector(x_h);

	// Upload to device
	GArray<float> x_dev, spmv_0, spmv_1;
	x_dev.upload(x_h);
	spmv_0.create(applier_0->MatrixSize());
	spmv_1.create(applier_1->MatrixSize());

	// Apply it
	applier_0->ApplySpMV(GArrayView<float>(x_dev), GArraySlice<float>(spmv_0));
	applier_1->ApplySpMV(GArrayView<float>(x_dev), GArraySlice<float>(spmv_1));

	// Download and check
	std::vector<float> spmv_0_h, spmv_1_h;
	spmv_0.download(spmv_0_h);
	spmv_1.download(spmv_1_h);
	LOG(INFO) << "The relative error between two spmv is " << maxRelativeError(spmv_0_h, spmv_1_h, 1e-3);
}