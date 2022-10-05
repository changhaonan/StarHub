#pragma once
#include <cuda_runtime_api.h>
#include <iostream>
#include <initializer_list>

/* Expanding unsigned char representation
*/
template<unsigned dim>
struct ucharX {
	__host__ __device__ ucharX() {
		for (auto i = 0; i < dim; ++i) {
			data[i] = 0;
		}
	}
	__host__ __device__ ucharX(const unsigned char* _data) {
		for (auto i = 0; i < dim; ++i) {
			data[i] = _data[i];
		}
	}
	// Assign API
	__host__ __device__ __forceinline__
		ucharX<dim>& operator=(const ucharX<dim>& other) {
#pragma unroll
		for (auto i = 0; i < dim; ++i) {
			data[i] = other[i];
		}
		return *this;
	}
	// Fetch API
	__host__ __device__ __forceinline__ unsigned char operator[](const unsigned pos) const {
		return data[pos];
	}
	__host__ __device__ __forceinline__ unsigned char& operator[](const unsigned pos) {
		return data[pos];
	}
	// Implict type cast
	__host__ __device__ operator const unsigned char* () const { return data; }
	__host__ __device__ operator unsigned char* () { return data; }

	// Data
	unsigned char data[dim];
};

template<unsigned dim>
__host__ __device__
ucharX<dim> make_ucharX(std::initializer_list<unsigned char> list) {
	ucharX<dim> vec;
	unsigned count = 0;
	for (auto v : list) {
		if (count >= dim) return vec;
		vec[count++] = v;
	}
	return vec;
}