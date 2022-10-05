#pragma once
#include <cuda_runtime.h>
#include "typeX.h"

template<template<unsigned> class T, unsigned dim>
__device__ __host__
auto min_val(const T<dim>& vec) -> decltype(vec.data[0]) {
	static_assert(dim != 0, "Dim for typeX can't be 0");
	auto min_val = vec[0];
	for (auto i = 1; i < dim; ++i) {
		if (min_val > vec[i]) {
			min_val = vec[i];
		}
	}
	return min_val;
}

template<template<unsigned> class T, unsigned dim>
__device__ __host__
auto max_val(const T<dim>& vec) -> decltype(vec.data[0]) {
	static_assert(dim != 0, "Dim for typeX can't be 0");
	auto max_val = vec[0];
	for (auto i = 1; i < dim; ++i) {
		if (max_val < vec[i]) {
			max_val = vec[i];
		}
	}
	return max_val;
}

template<template<unsigned> class T, unsigned dim>
__device__ __host__
int max_id(const T<dim>& vec) {
	static_assert(dim != 0, "Dim for typeX can't be 0");
	int id = 0;
	auto max_val = vec[0];
	for (auto i = 1; i < dim; ++i) {
		if (max_val < vec[i]) {
			max_val = vec[i];
			id = i;
		}
	}
	return id;
}