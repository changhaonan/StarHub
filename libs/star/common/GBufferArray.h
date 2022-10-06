#pragma once
#include <star/common/logging.h>
#include <star/common/ArrayView.h>
#include <star/common/ArraySlice.h>

namespace star {
	
	template<typename T>
	class GBufferArray {
	public:
		explicit GBufferArray() : m_buffer(nullptr, 0), m_array(nullptr, 0) {}
		explicit GBufferArray(size_t capacity) {
			AllocateBuffer(capacity);
			m_array = GArray<T>(m_buffer.ptr(), 0);
		}
		~GBufferArray() = default;

		// No implicit copy/assign/move
		STAR_NO_COPY_ASSIGN_MOVE(GBufferArray);

		// Accessing method
		GArray<T> Array() const { return m_array; }
		GArray<T> Buffer() const { return m_buffer; }
		GArrayView<T> View() const { return GArrayView<T>(m_array.ptr(), m_array.size()); }
		GArraySlice<T> Slice() { return GArraySlice<T>(m_array.ptr(), m_array.size()); }
		
		// The swap method
		void swap(GBufferArray<float>& other) {
			m_buffer.swap(other.m_buffer);
			m_array.swap(other.m_array);
		}
		
		// Cast to raw pointer
		const T* Ptr() const { return m_buffer.ptr(); }
		T* Ptr() { return m_buffer.ptr(); }
		operator T*() { return m_buffer.ptr(); }
		operator const T*() const { return m_buffer.ptr(); }
		
		// Query the size
		size_t BufferSize() const { return m_buffer.size(); }
		size_t BufferByteSize() const { return m_buffer.size() * sizeof(T); }
		size_t ArraySize() const { return m_array.size(); }
		size_t ArrayByteSize() const { return m_array.size() * sizeof(T); }

		// The allocation and changing method
		void AllocateBuffer(size_t capacity) {
			if(m_buffer.size() > capacity) return;
			m_buffer.create(capacity);
			m_array = GArray<T>(m_buffer.ptr(), 0);
		}
		void ReleaseBuffer() {
			if(m_buffer.size() > 0) m_buffer.release();
		}
		
		// Resize
		bool ResizeArray(size_t size, bool allocate = false) {
			if(size <= m_buffer.size()) {
				m_array = GArray<T>(m_buffer.ptr(), size);
				return true;
			} 
			else if(allocate) {
				const size_t prev_size = m_array.size();

				// Need to copy the old elements
				GArray<T> old_buffer = m_buffer;
				m_buffer.create(static_cast<size_t>(size * 1.5));
				if(prev_size > 0) {
					cudaSafeCall(cudaMemcpy(m_buffer.ptr(), old_buffer.ptr(), sizeof(T) * prev_size, cudaMemcpyDeviceToDevice));
					old_buffer.release();
				}

				// Correct the size
				m_array = GArray<T>(m_buffer.ptr(), prev_size);
				return true;
			} 
			else {
				return false;
			}
		}
		void ResizeArrayOrException(size_t size) {
			if (size > m_buffer.size()) {
				LOG(FATAL) << "The pre-allocated buffer is not enough: ask " << size << ", has " << m_buffer.size();
			}
			// Change the size of array
			m_array = GArray<T>(m_buffer.ptr(), size);
		}

	private:
		GArray<T> m_buffer;
		GArray<T> m_array;
	};
}