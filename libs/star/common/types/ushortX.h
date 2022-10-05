#pragma once
#include <cuda_runtime_api.h>
#include <iostream>
#include <initializer_list>

/* Expanding unsigned short representation
*/
template<unsigned dim>
struct ushortX {
	__host__ __device__ ushortX() {
		for (auto i = 0; i < dim; ++i) {
			data[i] = 0;
		}
	}
	__host__ __device__ ushortX(const unsigned short* _data) {
		for (auto i = 0; i < dim; ++i) {
			data[i] = _data[i];
		}
	}
	// Assign API
	__host__ __device__ __forceinline__ 
	ushortX<dim>& operator=(const ushortX<dim>& other) {
#pragma unroll
		for (auto i = 0; i < dim; ++i) {
			data[i] = other[i];
		}
		return *this;
	}
	// Fetch API
	__host__ __device__ __forceinline__ unsigned short operator[](const unsigned pos) const {
		return data[pos];
	}
	__host__ __device__ __forceinline__ unsigned short& operator[](const unsigned pos) {
		return data[pos];
	}
	// Implict type cast
	__host__ __device__ operator const unsigned short* () const { return data; }
	__host__ __device__ operator unsigned short* () { return data; }

	// Data
	unsigned short data[dim];
};

template<unsigned dim>
__host__ __device__
ushortX<dim> make_ushortX(std::initializer_list<unsigned short> list) {
	ushortX<dim> vec;
	unsigned count = 0;
	for (auto v : list) {
		if (count >= dim) return vec;
		vec[count++] = v;
	}
	return vec;
}