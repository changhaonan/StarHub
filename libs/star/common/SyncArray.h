#pragma once
#include <star/common/macro_utils.h>
#include <star/common/common_types.h>
#include <star/common/ArrayView.h>
#include <star/common/GBufferArray.h>

namespace star
{
	/**
	 * \brief The array with synchronize functionalities. Note that the content of the
	 *        array is not guarantee to be synchronized and need explict sync. However,
	 *        the array size of host and device array are always the same.
	 * \tparam T
	 */
	template <typename T>
	class SyncArray
	{
	public:
		explicit SyncArray() : m_host_array(), m_device_array() {}
		explicit SyncArray(size_t capacity)
		{
			AllocateBuffer(capacity);
		}
		~SyncArray() = default;
		STAR_NO_COPY_ASSIGN_MOVE(SyncArray);

		// The accessing interface
		size_t BufferSize() const { return m_device_array.BufferSize(); }
		size_t DeviceArraySize() const { return m_device_array.ArraySize(); }
		size_t HostArraySize() const { return m_host_array.size(); }

		GArrayView<T> DeviceArrayReadOnly() const { return GArrayView<T>(m_device_array.Array()); }
		star::GArray<T> GArray() const { return m_device_array.Array(); }
		GArraySlice<T> DeviceArrayReadWrite() { return m_device_array.Slice(); }

		std::vector<T> &HostArray() { return m_host_array; }
		const std::vector<T> &HostArray() const { return m_host_array; }

		// Access the raw pointer
		const T *DevicePtr() const { return m_device_array.Ptr(); }
		T *DevicePtr() { return m_device_array.Ptr(); }

		// The (possible) allocate interface
		void AllocateBuffer(size_t capacity)
		{
			m_host_array.reserve(capacity);
			m_device_array.AllocateBuffer(capacity);
		}

		// Release interface
		void ReleaseBuffer()
		{ // TODO: I added this, am I right?
			m_host_array.clear();
			m_device_array.ReleaseBuffer();
		}

		// the GBufferArray has implement resize with copy
		bool ResizeArray(size_t size, bool allocate = false)
		{
			if (m_device_array.ResizeArray(size, allocate) == true)
			{
				m_host_array.resize(size);
				return true;
			}

			// The device array can not resize success
			// The host and device are in the same size
			return false;
		}
		void ResizeArrayOrException(size_t size)
		{
			m_device_array.ResizeArrayOrException(size);
			m_host_array.resize(size);
		}

		// Clear the array of both host and device array
		// But DO NOT TOUCH the allocated buffer
		void ClearArray()
		{
			ResizeArray(0);
		}

		// The sync interface
		void SyncToDevice(cudaStream_t stream = 0)
		{
			// Update the size
			m_device_array.ResizeArrayOrException(m_host_array.size());

			// Actual sync
			cudaSafeCall(cudaMemcpyAsync(
				m_device_array.Ptr(),
				m_host_array.data(),
				sizeof(T) * m_host_array.size(),
				cudaMemcpyHostToDevice, stream));
		}
		void SyncToHost(cudaStream_t stream = 0, bool sync = true)
		{
			// Resize host array
			m_host_array.resize(m_device_array.ArraySize());

			// Actual sync
			cudaSafeCall(cudaMemcpyAsync(
				m_host_array.data(),
				m_device_array.Ptr(),
				sizeof(T) * m_host_array.size(),
				cudaMemcpyDeviceToHost, stream));

			if (sync)
			{
				// Before using on host, must call stream sync
				// But the call might be delayed
				cudaSafeCall(cudaStreamSynchronize(stream));
			}
		}

	private:
		std::vector<T> m_host_array;
		GBufferArray<T> m_device_array;
	};
}