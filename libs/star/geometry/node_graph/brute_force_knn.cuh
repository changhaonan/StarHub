#pragma once
#include "common/common_utils.h"
#include "star/common/global_configs.h"
#include <vector_types.h>


namespace star { namespace device {
	
	struct KnnHeapDevice {
		float4& distance;
		ushort4& index;
		
		// The constructor just copy the pointer, the class will modifiy it
		__host__ __device__ KnnHeapDevice(float4& dist, ushort4& node_idx) : distance(dist), index(node_idx) {}
		
		// The update function
		__host__ __device__ __forceinline__ 
		void update(unsigned short idx, float dist) {
			if (dist < distance.x) {
				distance.x = dist;
				index.x = idx;
				
				if (distance.y < distance.z) {
					if (distance.x < distance.z) {
						swap(distance.x, distance.z);
						swap(index.x, index.z);
					}
				}
				else {
					if (distance.x < distance.y) {
						swap(distance.x, distance.y);
						swap(index.x, index.y);
						if (distance.y < distance.w) {
							swap(distance.y, distance.w);
							swap(index.y, index.w);
						}
					}
				}
			}
		}
	};


	/*
	* \brief: Expand capcity to arbitrary size
	*/
	template<unsigned capacity>
	struct KnnHeapExpandDevice {
		float distance[capacity];
		unsigned short index[capacity];
		unsigned max_inner_id;
		float max_distance;

		// The constructor just copy the pointer, the class will modifiy it
		__host__ __device__ KnnHeapExpandDevice(float* __restrict__ dist, unsigned short* __restrict__ node_idx) {
			for (auto i = 0; i < capacity; ++i) {
				distance[i] = dist[i];
				index[i] = node_idx[i];
			}
			updateHeap();
		}

		// The update function
		__host__ __device__ __forceinline__
			void update(unsigned short idx, float dist) {
			if (dist < max_distance) {
				distance[max_inner_id] = dist;
				index[max_inner_id] = idx;
			}
			updateHeap();
		}

		// Heap matain functino
		__host__ __device__ __forceinline__ void updateHeap() {
			max_inner_id = 0;
			max_distance = distance[0];
			for (auto i = 1; i < capacity; ++i) {
				if (distance[i] > max_distance) {
					max_distance = distance[i];
					max_inner_id = i;
				}
			}
		}

		// Sort: distance increasing order
		__host__ __device__ void sort() {
			quickSort(0, capacity - 1);
		}

		__host__ __device__ void quickSort(int low, int high) {
			if (low < high) {
				/* pi is partitioning index, arr[pi] is now
				   at right place */
				int pi = partition(low, high);

				quickSort(low, pi - 1);  // Before pi
				quickSort(pi + 1, high); // After pi
			}
		}

		__host__ __device__ int partition(int low, int high) {
			float pivot = distance[high]; // pivot 
			int i = (low - 1); // Index of smaller element and indicates the right position of pivot found so far

			for (int j = low; j <= high - 1; j++)
			{
				// If current element is smaller than the pivot 
				if (distance[j] < pivot)
				{
					i++; // Increment index of smaller element 
					swap(distance[i], distance[j]);
					swap(index[i], index[j]);
				}
			}
			swap(distance[i + 1], distance[high]);
			swap(index[i + 1], index[high]);
			return i + 1;
		}
	};


	__device__ __forceinline__ void bruteForceSearch4Padded(
		const float4& vertex, const float4* nodes, unsigned node_num,
		float4& distance,
		ushort4& node_idx
	) {
		// Construct the heap
		KnnHeapDevice heap(distance, node_idx);

		// The brute force search
		const auto padded_node_num = ((node_num + 3) / 4) * 4;
		for (int k = 0; k < padded_node_num; k += 4) {
			// Compute the distance to each nodes
			const float4& v0 = nodes[k + 0];
			const float dx0 = vertex.x - v0.x;
			const float dy0 = vertex.y - v0.y;
			const float dz0 = vertex.z - v0.z;

			const float4& v1 = nodes[k + 1];
			const float dx1 = vertex.x - v1.x;
			const float dy1 = vertex.y - v1.y;
			const float dz1 = vertex.z - v1.z;

			const float4& v2 = nodes[k + 2];
			const float dx2 = vertex.x - v2.x;
			const float dy2 = vertex.y - v2.y;
			const float dz2 = vertex.z - v2.z;

			const float4& v3 = nodes[k + 3];
			const float dx3 = vertex.x - v3.x;
			const float dy3 = vertex.y - v3.y;
			const float dz3 = vertex.z - v3.z;

			const float dx0_sq = __fmul_rn(dx0, dx0);
			const float dx1_sq = __fmul_rn(dx1, dx1);
			const float dx2_sq = __fmul_rn(dx2, dx2);
			const float dx3_sq = __fmul_rn(dx3, dx3);

			const float dxy0_sq = __fmaf_rn(dy0, dy0, dx0_sq);
			const float dxy1_sq = __fmaf_rn(dy1, dy1, dx1_sq);
			const float dxy2_sq = __fmaf_rn(dy2, dy2, dx2_sq);
			const float dxy3_sq = __fmaf_rn(dy3, dy3, dx3_sq);

			const float dist_0 = __fmaf_rn(dz0, dz0, dxy0_sq);
			const float dist_1 = __fmaf_rn(dz1, dz1, dxy1_sq);
			const float dist_2 = __fmaf_rn(dz2, dz2, dxy2_sq);
			const float dist_3 = __fmaf_rn(dz3, dz3, dxy3_sq);
			// End of distance computation

			// Update of distance index
			heap.update(k + 0, dist_0);
			heap.update(k + 1, dist_1);
			heap.update(k + 2, dist_2);
			heap.update(k + 3, dist_3);
		}// End of iteration over all nodes
	}
	

	// This method is deprecated and should not be used in later code
	__device__ __forceinline__ void bruteForceSearch4Padded(
		const float4& vertex, const float4* nodes, unsigned node_num,
		float& d0, float& d1, float& d2, float& d3,
		unsigned short& i0, unsigned short& i1, unsigned short& i2, unsigned short& i3
	) {
		// The brute force search
		const auto padded_node_num = ((node_num + 3) / 4) * 4;
		for (int k = 0; k < padded_node_num; k += 4) {
			// Compute the distance to each nodes
			const float4& v0 = nodes[k + 0];
			const float dx0 = vertex.x - v0.x;
			const float dy0 = vertex.y - v0.y;
			const float dz0 = vertex.z - v0.z;

			const float4& v1 = nodes[k + 1];
			const float dx1 = vertex.x - v1.x;
			const float dy1 = vertex.y - v1.y;
			const float dz1 = vertex.z - v1.z;

			const float4& v2 = nodes[k + 2];
			const float dx2 = vertex.x - v2.x;
			const float dy2 = vertex.y - v2.y;
			const float dz2 = vertex.z - v2.z;

			const float4& v3 = nodes[k + 3];
			const float dx3 = vertex.x - v3.x;
			const float dy3 = vertex.y - v3.y;
			const float dz3 = vertex.z - v3.z;

			const float dx0_sq = __fmul_rn(dx0, dx0);
			const float dx1_sq = __fmul_rn(dx1, dx1);
			const float dx2_sq = __fmul_rn(dx2, dx2);
			const float dx3_sq = __fmul_rn(dx3, dx3);

			const float dxy0_sq = __fmaf_rn(dy0, dy0, dx0_sq);
			const float dxy1_sq = __fmaf_rn(dy1, dy1, dx1_sq);
			const float dxy2_sq = __fmaf_rn(dy2, dy2, dx2_sq);
			const float dxy3_sq = __fmaf_rn(dy3, dy3, dx3_sq);

			const float dist_0 = __fmaf_rn(dz0, dz0, dxy0_sq);
			const float dist_1 = __fmaf_rn(dz1, dz1, dxy1_sq);
			const float dist_2 = __fmaf_rn(dz2, dz2, dxy2_sq);
			const float dist_3 = __fmaf_rn(dz3, dz3, dxy3_sq);
			// End of distance computation

			// Update of distance index
			if (dist_0 < d0) {
				d0 = dist_0;
				i0 = k;

				if (d1 < d2) {
					if (d0 < d2) {
						swap(d0, d2);
						swap(i0, i2);
					}
				}
				else {
					if (d0 < d1) {
						swap(d0, d1);
						swap(i0, i1);
						if (d1 < d3) {
							swap(d1, d3);
							swap(i1, i3);
						}
					}
				}
			}

			if (dist_1 < d0) {
				d0 = dist_1;
				i0 = k + 1;

				if (d1 < d2) {
					if (d0 < d2) {
						swap(d0, d2);
						swap(i0, i2);
					}
				}
				else {
					if (d0 < d1) {
						swap(d0, d1);
						swap(i0, i1);
						if (d1 < d3) {
							swap(d1, d3);
							swap(i1, i3);
						}
					}
				}
			}

			if (dist_2 < d0) {
				d0 = dist_2;
				i0 = k + 2;

				if (d1 < d2) {
					if (d0 < d2) {
						swap(d0, d2);
						swap(i0, i2);
					}
				}
				else {
					if (d0 < d1) {
						swap(d0, d1);
						swap(i0, i1);
						if (d1 < d3) {
							swap(d1, d3);
							swap(i1, i3);
						}
					}
				}
			}

			if (dist_3 < d0) {
				d0 = dist_3;
				i0 = k + 3;

				if (d1 < d2) {
					if (d0 < d2) {
						swap(d0, d2);
						swap(i0, i2);
					}
				}
				else {
					if (d0 < d1) {
						swap(d0, d1);
						swap(i0, i1);
						if (d1 < d3) {
							swap(d1, d3);
							swap(i1, i3);
						}
					}
				}
			}
		}// End of iteration over all nodes
	}

} // namespace device
} // namespace star