#include <star/common/common_types.h>
#include <star/common/sanity_check.h>
#include <star/common/algorithm_types.cuh>
#include <assert.h>
#include <device_launch_parameters.h>

namespace star::device
{
	void checkCudaIndexTexture(
		cudaTextureObject_t vertex,
		cudaTextureObject_t normal,
		cudaTextureObject_t index,
		const GArray<float4> &vertex_array,
		const GArray<float4> &normal_array)
	{
	}

}

void star::checkPrefixSum()
{
	// The size of testing
	std::vector<unsigned> test_sizes;
	test_sizes.push_back(1000000);
	test_sizes.push_back(3000000);

	PrefixSum prefixsum;

	for (int j = 0; j < test_sizes.size(); j++)
	{
		// Construct the tests
		int test_size = test_sizes[j];
		std::vector<unsigned> in_array_host;
		in_array_host.resize(test_size);
		for (auto i = 0; i < in_array_host.size(); i++)
		{
			in_array_host[i] = rand() % 100;
		}

		// Upload it to device
		GArray<unsigned> in_array;
		in_array.upload(in_array_host);

		// Do inclusive prefixsum on device
		prefixsum.InclusiveSum(in_array);

		// Check the result
		std::vector<unsigned> prefixsum_array_host;
		prefixsum.valid_prefixsum_array.download(prefixsum_array_host);
		assert(prefixsum_array_host.size() == in_array_host.size());
		int sum = 0;
		for (auto i = 0; i < in_array_host.size(); i++)
		{
			sum += in_array_host[i];
			assert(sum == prefixsum_array_host[i]);
		}

		// Do exclusive sum on device and check
		prefixsum.ExclusiveSum(in_array);
		prefixsum.valid_prefixsum_array.download(prefixsum_array_host);
		sum = 0;
		for (auto i = 0; i < in_array_host.size(); i++)
		{
			assert(sum == prefixsum_array_host[i]);
			sum += in_array_host[i];
		}
	}
}

void star::checkKeyValueSort()
{
	// The vector of test size
	std::vector<int> test_sizes;
	test_sizes.push_back(1000000);
	test_sizes.push_back(3000000);

	// Construct the sorter
	KeyValueSort<int, int> kv_sorter;

	// Do testing
	for (auto j = 0; j < test_sizes.size(); j++)
	{
		int test_size = test_sizes[j];

		// Construct the inputs at host
		std::vector<int> key_in_host, value_in_host;
		key_in_host.resize(test_size);
		value_in_host.resize(test_size);
		for (auto i = 0; i < key_in_host.size(); i++)
		{
			key_in_host[i] = rand() % test_size;
			value_in_host[i] = key_in_host[i] + 10;
		}

		// Upload them to device
		GArray<int> key_in;
		key_in.upload(key_in_host);
		GArray<int> value_in;
		value_in.upload(value_in_host);

		// Sort it
		kv_sorter.Sort(key_in, value_in);

		// Download the result
		std::vector<int> key_sorted_host, value_sorted_host;
		kv_sorter.valid_sorted_key.download(key_sorted_host);
		kv_sorter.valid_sorted_value.download(value_sorted_host);

		// Check the result
		for (auto i = 0; i < key_sorted_host.size() - 1; i++)
		{
			assert(key_sorted_host[i] <= key_sorted_host[i + 1]);
			assert(value_sorted_host[i] == key_sorted_host[i] + 10);
		}
	}
}

void star::checkFlagSelection()
{
	std::vector<int> test_sizes;
	test_sizes.push_back(1000000);
	test_sizes.push_back(3000000);
	// The selector
	FlagSelection flag_selector;

	for (auto i = 0; i < test_sizes.size(); i++)
	{
		// Construct the test input
		int test_size = test_sizes[i];
		int num_selected = 0;
		std::vector<char> flag_host;
		flag_host.resize(test_size);
		for (auto j = 0; j < flag_host.size(); j++)
		{
			flag_host[j] = (char)(rand() % 2);
			assert(flag_host[j] >= 0);
			assert(flag_host[j] <= 1);
			if (flag_host[j] == 1)
				num_selected++;
		}

		// Perform selection
		GArray<char> flags;
		flags.upload(flag_host);
		flag_selector.Select(flags);

		// Check the result
		std::vector<int> selected_idx_host;
		flag_selector.valid_selected_idx.download(selected_idx_host);
		assert(selected_idx_host.size() == num_selected);
		for (auto j = 0; j < selected_idx_host.size(); j++)
		{
			assert(flag_host[selected_idx_host[j]] == 1);
		}
	}
}

void star::checkUniqueSelection()
{
	std::vector<int> test_sizes;
	test_sizes.push_back(1000000);
	test_sizes.push_back(3000000);
	// The selector
	UniqueSelection unique_selector;

	for (auto i = 0; i < test_sizes.size(); i++)
	{
		// Construct the test input
		int test_size = test_sizes[i];
		int num_selected = 0;
		std::vector<int> key_host;
		key_host.resize(test_size);
		for (auto j = 0; j < key_host.size(); j++)
		{
			key_host[j] = (int)(rand() % 200);
		}

		// Perform selection
		GArray<int> d_keys_in;
		d_keys_in.upload(key_host);
		unique_selector.Select(d_keys_in);

		// Check the result: the size shall be almost 200
		num_selected = unique_selector.valid_selected_element.size();
		assert(num_selected >= 198);
	}
}

#include <star/common/sanity_check.h>
#include <star/common/common_utils.h>
#include <star/common/encode_utils.h>
#include <random>
#include <time.h>

void star::fillRandomVector(std::vector<float> &vec)
{
	std::default_random_engine generator((unsigned int)time(NULL));
	std::uniform_real_distribution<float> distribution;
	for (auto i = 0; i < vec.size(); i++)
	{
		vec[i] = distribution(generator);
	}
}

void star::fillRandomVector(std::vector<unsigned int> &vec)
{
	std::default_random_engine generator((unsigned int)time(NULL));
	std::uniform_int_distribution<unsigned int> distribution;
	for (auto i = 0; i < vec.size(); i++)
	{
		vec[i] = distribution(generator);
	}
}

void star::test_encoding(const size_t test_size)
{
	std::default_random_engine generator((unsigned int)time(NULL));
	std::uniform_int_distribution<unsigned short> distribution;
	for (auto i = 0; i < test_size; i++)
	{
		// First encode it
		unsigned char r, g, b, a;
		r = distribution(generator);
		g = distribution(generator);
		b = distribution(generator);
		a = distribution(generator);
		float encoded = float_encode_rgba(r, g, b, a);
		// Next decode it
		unsigned char decode_r, decode_g, decode_b, decode_a;
		float_decode_rgba(encoded, decode_r, decode_g, decode_b, decode_a);
		assert(r == decode_r);
		assert(g == decode_g);
		assert(b == decode_b);
		assert(a == decode_a);
	}
}

void star::fillZeroVector(std::vector<float> &vec)
{
	for (auto i = 0; i < vec.size(); i++)
	{
		vec[i] = 0.0f;
	}
}

double star::maxRelativeError(
	const std::vector<float> &vec_0,
	const std::vector<float> &vec_1,
	const float small_cutoff)
{
	auto max_relaive_err = 0.0f;
	for (auto j = 0; j < std::min(vec_0.size(), vec_1.size()); j++)
	{
		float value_0 = vec_0[j];
		float value_1 = vec_1[j];
		if (std::isnan(value_0))
		{
			LOG(INFO) << "Found Nan value at first element " << j << " !";
		}
		if (std::isnan(value_1))
		{
			LOG(INFO) << "Found Nan value at second element " << j << " !";
		}
		float err = std::abs(value_0 - value_1);
		if (err > small_cutoff)
		{
			float relative_err = std::abs(err / std::max(std::abs(value_0), std::abs(value_1)));
			if (relative_err > max_relaive_err)
			{
				max_relaive_err = relative_err;
			}
			if (relative_err > 1e-3)
			{
				LOG(INFO) << "The relative error for " << j << " element is " << relative_err << " between " << vec_0[j] << " and " << vec_1[j];
			}
		}
	}
	return max_relaive_err;
}

double star::maxRelativeError(
	const std::vector<float> &vec_0,
	const std::vector<float> &vec_1,
	const float small_cutoff,
	const bool log_output,
	const size_t log_start,
	const size_t log_end)
{
	auto max_relaive_err = 0.0f;
	for (auto j = std::max(size_t(0), log_start); j < std::min(std::min(vec_0.size(), vec_1.size()), log_end); j++)
	{
		float value_0 = vec_0[j];
		float value_1 = vec_1[j];
		if (std::isnan(value_0))
		{
			LOG(INFO) << "Found Nan value at first element " << j << " !";
		}
		if (std::isnan(value_1))
		{
			LOG(INFO) << "Found Nan value at second element " << j << " !";
		}
		float err = std::abs(value_0 - value_1);
		if (err > small_cutoff)
		{
			float relative_err = std::abs(err / std::max(std::abs(value_0), std::abs(value_1)));
			if (relative_err > max_relaive_err)
			{
				max_relaive_err = relative_err;
			}
			if (relative_err > 1e-3)
			{
				if (log_output)
					LOG(INFO) << "The relative error for " << j << " element is " << relative_err << " between " << vec_0[j] << " and " << vec_1[j];
			}
		}
	}
	return max_relaive_err;
}

double star::maxRelativeError(const GArray<float> &vec_0, const GArray<float> &vec_1, const float small_cutoff)
{
	std::vector<float> h_vec_0, h_vec_1;
	vec_0.download(h_vec_0);
	vec_1.download(h_vec_1);
	return maxRelativeError(h_vec_0, h_vec_1, small_cutoff);
}

void star::randomShuffle(std::vector<unsigned int> &key, std::vector<unsigned int> &value)
{
	std::srand(time(nullptr));
	for (auto i = key.size() - 1; i > 0; --i)
	{
		const auto swap_idx = std::rand() % (i + 1);
		std::swap(key[i], key[swap_idx]);
		std::swap(value[i], value[swap_idx]);
		// rand() % (i+1) isn't actually correct, because the generated number
		// is not uniformly distributed for most values of i. A correct implementation
		// will need to essentially reimplement C++11 std::uniform_int_distribution,
		// which is beyond the scope of this example.
	}
}

void star::randomShuffle(std::vector<unsigned> &vec)
{
	std::srand(time(nullptr));
	for (auto i = vec.size() - 1; i > 0; --i)
	{
		const auto swap_idx = std::rand() % (i + 1);
		std::swap(vec[i], vec[swap_idx]);
		// rand() % (i+1) isn't actually correct, because the generated number
		// is not uniformly distributed for most values of i. A correct implementation
		// will need to essentially reimplement C++11 std::uniform_int_distribution,
		// which is beyond the scope of this example.
	}
}

void star::fillMultiKeyValuePairs(
	std::vector<unsigned int> &h_keys,
	const unsigned int num_entries,
	const unsigned int key_maximum,
	const unsigned int average_duplication)
{
	// The random number generator
	std::default_random_engine generator((unsigned int)time(NULL));
	std::uniform_int_distribution<int> distribution;

	// Insert into the key array
	h_keys.clear();
	int remains_entries = num_entries;
	while (remains_entries > 0)
	{
		auto key = distribution(generator) % key_maximum;
		auto key_duplication = distribution(generator) % (2 * average_duplication);
		if (key_duplication > remains_entries)
			key_duplication = remains_entries;
		remains_entries -= key_duplication;
		for (auto i = 0; i < key_duplication; i++)
		{
			h_keys.push_back(key);
		}
	}

	// Perform a random shuffle
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(h_keys.begin(), h_keys.end(), g);
}

bool star::isElementUnique(const std::vector<unsigned> &vec, const unsigned empty)
{
	std::vector<unsigned> sorted_vec(vec);
	std::sort(sorted_vec.begin(), sorted_vec.end());
	for (auto i = 0; i < sorted_vec.size() - 1; i++)
	{
		if (sorted_vec[i] == sorted_vec[i + 1] && sorted_vec[i] != empty)
		{
			return false;
		}
	}
	return true;
}

unsigned star::numUniqueElement(const std::vector<unsigned> &vec, const unsigned empty)
{
	// First do a sorting
	std::vector<unsigned> sorted_vec(vec);
	std::sort(sorted_vec.begin(), sorted_vec.end());

	// Count the elements
	unsigned unique_elements = 1; // The first element is assumed to be unique
	for (auto i = 1; i < sorted_vec.size(); i++)
	{
		if (sorted_vec[i] == sorted_vec[i - 1] || sorted_vec[i] == empty)
		{
			// duplicate or invalid elements
		}
		else
		{
			unique_elements++;
		}
	}

	return unique_elements;
}

unsigned star::numUniqueElement(const star::GArray<unsigned int> &vec, const unsigned empty)
{
	std::vector<unsigned> h_vec;
	vec.download(h_vec);
	return numUniqueElement(h_vec, empty);
}

unsigned star::numNonZeroElement(const std::vector<unsigned> &vec)
{
	unsigned nonzero_count = 0;
	for (auto i = 0; i < vec.size(); i++)
	{
		if (vec[i] != 0)
		{
			nonzero_count++;
		}
	}
	return nonzero_count;
}

unsigned star::numNonZeroElement(const GArray<unsigned int> &vec)
{
	std::vector<unsigned> h_vec;
	vec.download(h_vec);
	return numNonZeroElement(h_vec);
}

unsigned star::numNonZeroElement(const star::GArrayView<unsigned int> &vec)
{
	std::vector<unsigned> h_vec;
	vec.Download(h_vec);
	return numNonZeroElement(h_vec);
}

bool star::containsNaN(const std::vector<float4> &vec)
{
	for (auto i = 0; i < vec.size(); i++)
	{
		const float4 element = vec[i];
		if (std::isnan(element.x))
			return true;
		if (std::isnan(element.y))
			return true;
		if (std::isnan(element.z))
			return true;
		if (std::isnan(element.w))
			return true;
	}
	return false;
}

bool star::containsNaN(const star::GArrayView<float4> &vec)
{
	std::vector<float4> h_vec;
	vec.Download(h_vec);
	return containsNaN(h_vec);
}

void star::applyRandomSE3ToWarpField(
	std::vector<DualQuaternion> &node_se3,
	float max_rot, float max_trans)
{
	std::default_random_engine generator((unsigned int)time(NULL));
	std::uniform_real_distribution<float> rot_distribution(-max_rot, max_rot);
	std::uniform_real_distribution<float> trans_distribution(-max_trans, max_trans);
	for (auto i = 0; i < node_se3.size(); i++)
	{
		float3 twist_rot, twist_trans;
		twist_rot.x = rot_distribution(generator);
		twist_rot.y = rot_distribution(generator);
		twist_rot.z = rot_distribution(generator);
		twist_trans.x = trans_distribution(generator);
		twist_trans.y = trans_distribution(generator);
		twist_trans.z = trans_distribution(generator);
		apply_twist(twist_rot, twist_trans, node_se3[i]);
	}
}

void star::applyRandomSE3ToWarpField(
	GArraySlice<DualQuaternion> node_se3,
	float max_rot, float max_trans)
{
	std::vector<DualQuaternion> h_node_se3;
	node_se3.DownloadSync(h_node_se3);
	applyRandomSE3ToWarpField(h_node_se3, max_rot, max_trans);
	node_se3.UploadSync(h_node_se3);
}

double star::differenceL2(const std::vector<float> &vec_0, const std::vector<float> &vec_1)
{
	double diff_square = 0.0;
	for (auto i = 0; i < std::min(vec_0.size(), vec_1.size()); i++)
	{
		const auto diff = vec_0[i] - vec_1[i];
		diff_square += (diff * diff);
	}
	return std::sqrt(diff_square);
}

double star::differenceL2(const GArray<float> &vec_0, const GArray<float> &vec_1)
{
	std::vector<float> h_vec_0, h_vec_1;
	vec_0.download(h_vec_0);
	vec_1.download(h_vec_1);
	return differenceL2(h_vec_0, h_vec_1);
}

// The statistic method about any residual vector, the value might be negative
void star::residualVectorStatistic(const std::vector<float> &residual_in, int topk, std::ostream &output)
{
	// Apply abs to all input
	std::vector<float> sorted_residual_vec;
	sorted_residual_vec.clear();
	sorted_residual_vec.resize(residual_in.size());

	double average_residual = 0.0;
	for (auto i = 0; i < residual_in.size(); i++)
	{
		sorted_residual_vec[i] = std::abs(residual_in[i]);
		average_residual += sorted_residual_vec[i];
	}

	output << "The average of residual is " << average_residual / residual_in.size() << std::endl;

	// Sort it
	std::sort(sorted_residual_vec.begin(), sorted_residual_vec.end());

	// Max, min and medium
	const auto mid_idx = residual_in.size() >> 1;
	output << "The max, middle and min of residual is "
		   << sorted_residual_vec[sorted_residual_vec.size() - 1]
		   << " " << sorted_residual_vec[mid_idx]
		   << " " << sorted_residual_vec[0] << std::endl;

	// The top k residual
	output << "The top " << topk << " residual is";
	for (auto i = 0; i < topk; i++)
	{
		int idx = sorted_residual_vec.size() - 1 - i;
		if (idx >= 0 && idx < sorted_residual_vec.size())
		{
			output << " " << sorted_residual_vec[idx];
		}
	}
	output << std::endl;
}