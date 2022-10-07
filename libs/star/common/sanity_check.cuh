#pragma once
#include <vector_types.h>
#include <texture_types.h>
#include <ostream>
#include <iostream>
#include <star/common/common_types.h>
#include <star/common/ArraySlice.h>
#include <star/math/DualQuaternion.hpp>

namespace star
{

	// Debug and Check the result of algorithm-types
	void checkPrefixSum();
	void checkKeyValueSort();
	void checkFlagSelection();
	void checkUniqueSelection();

	// Test of encoding
	void test_encoding(const size_t test_size = 10000);

	// Fill random arrays
	void fillRandomVector(std::vector<float> &vec);
	void fillRandomVector(std::vector<unsigned int> &vec);
	void fillZeroVector(std::vector<float> &vec);
	void randomShuffle(std::vector<unsigned int> &key, std::vector<unsigned int> &value);
	void randomShuffle(std::vector<unsigned> &vec);

	void fillMultiKeyValuePairs(
		std::vector<unsigned int> &h_keys,
		const unsigned int num_entries,
		const unsigned int key_maximum,
		const unsigned int average_duplication = 32);

	// Check the uniqueness of a vector, do not count empty
	bool isElementUnique(const std::vector<unsigned> &vec, const unsigned empty);
	unsigned numUniqueElement(const std::vector<unsigned> &vec, const unsigned empty);
	unsigned numUniqueElement(const GArray<unsigned> &vec, const unsigned empty);

	// The number of non-zero elements
	unsigned numNonZeroElement(const std::vector<unsigned> &vec);
	unsigned numNonZeroElement(const GArray<unsigned> &vec);
	unsigned numNonZeroElement(const GArrayView<unsigned> &vec);

	// Does the vector contains NaN
	bool containsNaN(const std::vector<float4> &vec);
	bool containsNaN(const GArrayView<float4> &vec);

	// The random init of warp field
	void applyRandomSE3ToWarpField(std::vector<DualQuaternion> &node_se3, float max_rot = 0.1, float max_trans = 0.1);
	void applyRandomSE3ToWarpField(GArraySlice<DualQuaternion> node_se3, float max_rot = 0.1, float max_trans = 0.1);

	template <typename T>
	bool isElementUniqueNaive(const std::vector<T> &vec, const T empty);
	template <typename T>
	bool isElementUniqueNonEmptyNaive(const std::vector<T> &vec, const T empty);

	template <typename T>
	double averageDuplicate(const std::vector<T> &vec, const T empty);

	// Compute the maximun relative error, the two input
	// Vectors are expected to match
	double maxRelativeError(const std::vector<float> &vec_0, const std::vector<float> &vec_1, const float small_cutoff,
							const bool log_output, const size_t log_start, const size_t log_end);
	double maxRelativeError(const std::vector<float> &vec_0, const std::vector<float> &vec_1, const float small_cutoff);
	double maxRelativeError(const GArray<float> &vec_0, const GArray<float> &vec_1, const float small_cutoff);

	// Compute the L2 different norm
	// The two input are expected to match
	double differenceL2(const std::vector<float> &vec_0, const std::vector<float> &vec_1);
	double differenceL2(const GArray<float> &vec_0, const GArray<float> &vec_1);

	// The statistic about residual error, the input might be negative
	void residualVectorStatistic(const std::vector<float> &residual_vec, int topk = 10, std::ostream &output = std::cout);

	namespace device
	{
		// Check whether the texture is correct
		void checkCudaIndexTexture(
			cudaTextureObject_t vertex,
			cudaTextureObject_t normal,
			cudaTextureObject_t index_texture,
			const GArray<float4> &vertex_array,
			const GArray<float4> &normal_array);

	} // End of namespace device

} // End of star

// The implementation file
#include <star/common/sanity_check.hpp>