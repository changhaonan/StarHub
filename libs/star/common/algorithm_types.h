#pragma once
#include <star/common/common_types.h>
#include <star/common/ArrayView.h>

namespace star
{

	/* Small struct to hold the storage and result for prefix sum
	   Should be allocated and used locally
	*/
	struct PrefixSum
	{
	private:
		// Shared buffer, not thread safe
		GArray<unsigned char> m_temp_storage;
		GArray<unsigned> m_prefixsum_buffer;

	public:
		// Functional interface
		__host__ void AllocateBuffer(size_t input_size);

		__host__ void InclusiveSum(const GArray<unsigned> &array_in, cudaStream_t stream = 0, bool debug_sync = false);
		__host__ void InclusiveSum(const GArrayView<unsigned> &array_in, cudaStream_t stream = 0);

		__host__ void ExclusiveSum(const GArray<unsigned> &array_in, cudaStream_t stream = 0, bool debug_sync = false);

		// The pointer of result
		GArray<unsigned> valid_prefixsum_array;
	};

	/* Small struct for key-value sorting
	 */
	template <typename KeyT, typename ValueT>
	struct KeyValueSort
	{
	private:
		// Shared buffer
		GArray<unsigned char> m_temp_storage;
		GArray<KeyT> m_sorted_key_buffer;
		GArray<ValueT> m_sorted_value_buffer;

	public:
		void AllocateBuffer(size_t input_size);

		void Sort(
			const GArray<KeyT> &key_in,
			const GArray<ValueT> &value_in,
			cudaStream_t stream = 0,
			int end_bit = sizeof(KeyT) * 8,
			bool debug_sync = false);

		void Sort(const GArrayView<KeyT> &key_in, const GArrayView<ValueT> &value_in, cudaStream_t stream = 0);
		void Sort(const GArrayView<KeyT> &key_in, const GArrayView<ValueT> &value_in, int end_bit, cudaStream_t stream = 0);

		// Sort key only
		void Sort(const GArray<KeyT> &key_in,
				  cudaStream_t stream = 0,
				  int end_bit = sizeof(KeyT) * 8,
				  bool debug_sync = false);

		// Sorted value
		GArray<KeyT> valid_sorted_key;
		GArray<ValueT> valid_sorted_value;
	};

	/*
	 * The struct for flag selection
	 */
	struct FlagSelection
	{
	private:
		GArray<int> m_selection_input_buffer;
		GArray<int> m_selected_idx_buffer;
		GArray<unsigned char> m_temp_storage;

		// The memory for the number of selected index
		int *m_device_num_selected;
		int *m_host_num_selected;

	public:
		FlagSelection()
		{
			cudaSafeCall(cudaMalloc((void **)(&m_device_num_selected), sizeof(int)));
			cudaSafeCall(cudaMallocHost((void **)(&m_host_num_selected), sizeof(int)));
		}

		~FlagSelection()
		{
			cudaSafeCall(cudaFree(m_device_num_selected));
			cudaSafeCall(cudaFreeHost(m_host_num_selected));
		}

		// Function interface
		void AllocateAndInit(size_t input_size, cudaStream_t stream = 0);

		// When the element is not defined, do selection on index
		void Select(const GArray<char> &flags, cudaStream_t stream = 0, bool debug_sync = false);
		// The output: pointer to m_selected_idx_buffer
		GArray<int> valid_selected_idx;

		// When the selected element is just unsigned array
		void SelectUnsigned(
			const GArray<char> &flags,
			const GArray<unsigned> &select_from,
			GArray<unsigned> &select_to_buffer,
			GArray<unsigned> &valid_select_to_array,
			cudaStream_t stream = 0);

		// The buffer for flag
		GArray<char> select_indicator_buffer;
	};

	/*
	 * The struct for unique selection
	 */
	struct UniqueSelection
	{
	private:
		GArray<int> m_selected_element_buffer;
		GArray<unsigned char> m_temp_storage;

		// The memory for the number of selected index
		int *m_device_num_selected;
		int *m_host_num_selected;

	public:
		UniqueSelection()
		{
			cudaSafeCall(cudaMalloc((void **)(&m_device_num_selected), sizeof(int)));
			cudaSafeCall(cudaMallocHost((void **)(&m_host_num_selected), sizeof(int)));
		}
		~UniqueSelection()
		{
			cudaSafeCall(cudaFree(m_device_num_selected));
			cudaSafeCall(cudaFreeHost(m_host_num_selected));
		}

		void Allocate(size_t input_size);
		void Select(const GArray<int> &key_in, cudaStream_t stream = 0, bool debug_sync = false);

		// The output is the selected element as a pointer to above
		GArray<int> valid_selected_element;
	};
}

#if defined(__CUDACC__)
#include <star/common/algorithm_types.cuh>
#endif