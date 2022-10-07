#pragma once
#include <star/common/algorithm_types.h>
#include <cub/cub.cuh>

template <typename KeyT, typename ValueT>
void star::KeyValueSort<KeyT, ValueT>::AllocateBuffer(size_t input_size)
{
	if (input_size <= m_sorted_key_buffer.size())
		return;

	// If exists buffer, clear them
	if (m_sorted_key_buffer.size() > 0)
	{
		m_sorted_key_buffer.release();
		m_sorted_value_buffer.release();
		m_temp_storage.release();
	}

	// Allocate buffer
	size_t allocate_size = 3 * input_size / 2;
	m_sorted_key_buffer.create(allocate_size);
	m_sorted_value_buffer.create(allocate_size);

	// Query the required temp storage
	size_t required_temp_bytes = 0;
	cub::DeviceRadixSort::SortPairs<KeyT, ValueT>(
		NULL, required_temp_bytes,
		m_sorted_key_buffer.ptr(), m_sorted_key_buffer.ptr(),
		m_sorted_value_buffer.ptr(), m_sorted_value_buffer.ptr(),
		(int)m_sorted_key_buffer.size());

	// Allocate the required storage
	m_temp_storage.create(required_temp_bytes);
}

template <typename KeyT, typename ValueT>
void star::KeyValueSort<KeyT, ValueT>::Sort(
	const GArray<KeyT> &key_in,
	const GArray<ValueT> &value_in,
	cudaStream_t stream, int end_bit, bool debug_sync)
{
	// Allocate the buffer if not enough
	AllocateBuffer(key_in.size());

	// Construct the correct-sized result
	valid_sorted_key = GArray<KeyT>(m_sorted_key_buffer.ptr(), key_in.size());
	valid_sorted_value = GArray<ValueT>(m_sorted_value_buffer.ptr(), value_in.size());

	// Do sorting
	size_t required_temp_bytes = m_temp_storage.sizeBytes();
	cub::DeviceRadixSort::SortPairs<KeyT, ValueT>(
		m_temp_storage.ptr(), required_temp_bytes,
		key_in.ptr(), valid_sorted_key.ptr(),
		value_in.ptr(), valid_sorted_value.ptr(),
		(int)key_in.size(),
		0, end_bit,
		stream, debug_sync);
}

template <typename KeyT, typename ValueT>
void star::KeyValueSort<KeyT, ValueT>::Sort(
	const star::GArrayView<KeyT> &key_in,
	const star::GArrayView<ValueT> &value_in,
	cudaStream_t stream)
{
	// Allocate the buffer if not enough
	AllocateBuffer(key_in.Size());

	// Construct the correct-sized result
	valid_sorted_key = GArray<KeyT>(m_sorted_key_buffer.ptr(), key_in.Size());
	valid_sorted_value = GArray<ValueT>(m_sorted_value_buffer.ptr(), value_in.Size());

	// Do sorting
	size_t required_temp_bytes = m_temp_storage.sizeBytes();
	cub::DeviceRadixSort::SortPairs<KeyT, ValueT>(
		m_temp_storage.ptr(), required_temp_bytes,
		key_in.Ptr(), valid_sorted_key.ptr(),
		value_in.Ptr(), valid_sorted_value.ptr(),
		(int)key_in.Size(),
		0, 8 * sizeof(KeyT),
		stream, false);
}

template <typename KeyT, typename ValueT>
void star::KeyValueSort<KeyT, ValueT>::Sort(
	const star::GArrayView<KeyT> &key_in,
	const star::GArrayView<ValueT> &value_in,
	int end_bit,
	cudaStream_t stream)
{
	// Allocate the buffer if not enough
	AllocateBuffer(key_in.Size());

	// Construct the correct-sized result
	valid_sorted_key = GArray<KeyT>(m_sorted_key_buffer.ptr(), key_in.Size());
	valid_sorted_value = GArray<ValueT>(m_sorted_value_buffer.ptr(), value_in.Size());

	// Do sorting
	size_t required_temp_bytes = m_temp_storage.sizeBytes();
	cub::DeviceRadixSort::SortPairs<KeyT, ValueT>(
		m_temp_storage.ptr(), required_temp_bytes,
		key_in.Ptr(), valid_sorted_key.ptr(),
		value_in.Ptr(), valid_sorted_value.ptr(),
		(int)key_in.Size(),
		0, end_bit,
		stream, false);
}

template <typename KeyT, typename ValueT>
void star::KeyValueSort<KeyT, ValueT>::Sort(
	const star::GArray<KeyT> &key_in,
	cudaStream_t stream,
	int end_bit,
	bool debug_sync)
{
	// Allocate the buffer if not enough
	AllocateBuffer(key_in.size());

	// Construct the correct-sized result
	valid_sorted_key = GArray<KeyT>(m_sorted_key_buffer.ptr(), key_in.size());

	// Invoke the sorter
	size_t required_temp_bytes = m_temp_storage.sizeBytes();
	cub::DeviceRadixSort::SortKeys(
		m_temp_storage.ptr(), required_temp_bytes,
		key_in.ptr(), valid_sorted_key.ptr(), key_in.size(),
		0, end_bit,
		stream,
		debug_sync);
}
