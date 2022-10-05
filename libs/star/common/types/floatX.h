#pragma once
#include <cuda_runtime_api.h>
#include <iostream>
#include <initializer_list>

/* Expanding float representation
*/
template<unsigned dim>
struct floatX {
	__host__ __device__ floatX() {
		for (auto i = 0; i < dim; ++i) {
			data[i] = 0.f;
		}
	}
	__host__ __device__ floatX(const float* _data) {
		for (auto i = 0; i < dim; ++i) {
			data[i] = _data[i];
		}
	}
	// Assign API
	__host__ __device__ __forceinline__
	floatX<dim>& operator=(const floatX<dim>& other) {
#pragma unroll
		for (auto i = 0; i < dim; ++i) {
			data[i] = other[i];
		}
		return *this;
	}
	// Operator
	__host__ __device__ __forceinline__
		floatX<dim> operator*(const float& scale) {
		floatX<dim> timed_val;
#pragma unroll
		for (auto i = 0; i < dim; ++i) {
			timed_val[i] = data[i] * scale;
		}
		return timed_val;
	}
	// Fetch API
	__host__ __device__ __forceinline__ float operator[](const unsigned pos) const {
		return data[pos];
	}
	__host__ __device__ __forceinline__ float& operator[](const unsigned pos) {
		return data[pos];
	}
	// Implict type cast
	__host__ __device__ operator const float* () const { return data; }
	__host__ __device__ operator float* () { return data; }

	// Data
	float data[dim];
};

template<unsigned dim>
__host__ __device__
floatX<dim> make_floatX(std::initializer_list<float> list) {
	floatX<dim> vec;
	unsigned count = 0;
	for (auto v : list) {
		if (count >= dim) return vec;
		vec[count++] = v;
	}
	return vec;
}