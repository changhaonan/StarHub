#pragma once
#include <vector_types.h>
#include <star/common/ArrayView.h>
#include "global_configs.h"

namespace star
{

	namespace TermScalarSize
	{
		constexpr unsigned DenseImage = 3; // 3 channel: 1 picp + 2 opticalflow
		constexpr unsigned Reg = 3;
		constexpr unsigned NodeTranslation = 3;
		constexpr unsigned Feature = 1;
		constexpr unsigned Invalid = 0;
	}

	namespace TermPairSize
	{
		constexpr unsigned DenseImage = d_surfel_knn_pair_size;
		constexpr unsigned Reg = 1;
		constexpr unsigned NodeTranslation = 0; // FIXME: Used as 0 for now
		constexpr unsigned Feature = d_surfel_knn_pair_size;
		constexpr unsigned Invalid = 0;
	}

	// Size & Offset & Occupation
	constexpr unsigned num_term_type = 4;
	enum class TermType
	{
		DenseImage = 0,
		Reg = 1,
		NodeTranslation = 2,
		Feature = 3,
		Invalid = 4
	};

	struct TermTypeOffset
	{
		unsigned offset_value[num_term_type];

		// The accessed interface
		__host__ __device__ __forceinline__ const unsigned &operator[](const int idx) const
		{
			return offset_value[idx];
		}

		// The size of terms
		__host__ __device__ __forceinline__ unsigned TermSize() const { return offset_value[num_term_type - 1]; }
		__host__ __forceinline__ unsigned ScalarTermSize() const
		{
			return DenseImageTermSize() + RegTermSize() + NodeTranslationTermSize() + FeatureTermSize();
		}

		// The different type of terms
		__host__ __device__ __forceinline__ unsigned DenseImageTermSize() const { return offset_value[0]; }
		__host__ __device__ __forceinline__ unsigned RegTermSize() const { return offset_value[1] - offset_value[0]; }
		__host__ __device__ __forceinline__ unsigned NodeTranslationTermSize() const { return offset_value[2] - offset_value[1]; }
		__host__ __device__ __forceinline__ unsigned FeatureTermSize() const { return offset_value[3] - offset_value[2]; }
	};

	inline void size2offset(
		TermTypeOffset &offset,
		unsigned num_dense_pixel_pair,
		unsigned num_node_pair,
		unsigned num_nodes,
		unsigned num_feature_pair)
	{
		unsigned prefixsum = 0;
		// Dense
		prefixsum += num_dense_pixel_pair;
		offset.offset_value[0] = prefixsum;
		// Reg
		prefixsum += num_node_pair;
		offset.offset_value[1] = prefixsum;
		// NodeTranslation
		prefixsum += num_nodes;
		offset.offset_value[2] = prefixsum;
		// Feature
		prefixsum += num_feature_pair;
		offset.offset_value[3] = prefixsum;
	}

	__host__ __device__ __forceinline__ void query_typed_index(unsigned term_idx, const TermTypeOffset &offset, TermType &type, unsigned &typed_idx)
	{
		unsigned term_size = 0;
		if (term_idx < term_size + offset.DenseImageTermSize())
		{
			type = TermType::DenseImage;
			typed_idx = term_idx - term_size;
			return;
		}
		term_size += offset.DenseImageTermSize();
		if (term_idx >= term_size && term_idx < term_size + offset.RegTermSize())
		{
			type = TermType::Reg;
			typed_idx = term_idx - term_size;
			return;
		}
		term_size += offset.RegTermSize();
		if (term_idx >= term_size && term_idx < term_size + offset.NodeTranslationTermSize())
		{
			type = TermType::NodeTranslation;
			typed_idx = term_idx - term_size;
			return;
		}
		term_size += offset.NodeTranslationTermSize();
		if (term_idx >= term_size && term_idx < term_size + offset.FeatureTermSize())
		{
			type = TermType::Feature;
			typed_idx = term_idx - term_size;
			return;
		}

		// Not a valid term
		type = TermType::Invalid;
		typed_idx = 0xFFFFFFFF;
	}

	__host__ __device__ __forceinline__ void query_typed_index(unsigned term_idx, const TermTypeOffset &offset, TermType &type, unsigned &typed_idx, unsigned &scalar_term_idx)
	{
		unsigned term_size = 0;
		unsigned scalar_offset = 0;
		if (term_idx < term_size + offset.DenseImageTermSize())
		{
			type = TermType::DenseImage;
			typed_idx = term_idx - term_size;
			scalar_term_idx = scalar_offset + term_idx * TermScalarSize::DenseImage;
			return;
		}
		term_size += offset.DenseImageTermSize();
		scalar_offset += offset.DenseImageTermSize() * TermScalarSize::DenseImage;
		if (term_idx >= term_size && term_idx < term_size + offset.RegTermSize())
		{
			type = TermType::Reg;
			typed_idx = term_idx - term_size;
			scalar_term_idx = scalar_offset + typed_idx * TermScalarSize::Reg;
			return;
		}
		term_size += offset.RegTermSize();
		scalar_offset += offset.RegTermSize() * TermScalarSize::Reg;
		if (term_idx >= term_size && term_idx < term_size + offset.NodeTranslationTermSize())
		{
			type = TermType::NodeTranslation;
			typed_idx = term_idx - term_size;
			scalar_term_idx = scalar_offset + typed_idx * TermScalarSize::NodeTranslation;
			return;
		}
		term_size += offset.NodeTranslationTermSize();
		scalar_offset += offset.NodeTranslationTermSize() * TermScalarSize::NodeTranslation;
		if (term_idx >= term_size && term_idx < term_size + offset.FeatureTermSize())
		{
			type = TermType::Feature;
			typed_idx = term_idx - term_size;
			scalar_term_idx = scalar_offset + typed_idx * TermScalarSize::Feature;
			return;
		}

		// Not a valid term
		type = TermType::Invalid;
		typed_idx = 0xFFFFFFFF;
		scalar_term_idx = 0xFFFFFFFF;
	}

	__host__ __device__ __forceinline__ void query_nodepair_index(unsigned term_idx, const TermTypeOffset &offset, TermType &type, unsigned &typed_idx, unsigned &nodepair_idx)
	{
		unsigned term_size = 0;
		unsigned pair_offset = 0;
		if (term_idx < term_size + offset.DenseImageTermSize())
		{
			type = TermType::DenseImage;
			typed_idx = term_idx - term_size;
			nodepair_idx = pair_offset + typed_idx * TermPairSize::DenseImage;
			return;
		}
		term_size += offset.DenseImageTermSize();
		pair_offset += offset.DenseImageTermSize() * TermPairSize::DenseImage;
		if (term_idx >= term_size && term_idx < term_size + offset.RegTermSize())
		{
			type = TermType::Reg;
			typed_idx = term_idx - term_size;
			nodepair_idx = pair_offset + typed_idx * TermPairSize::Reg;
			return;
		}
		term_size += offset.RegTermSize();
		pair_offset = pair_offset + offset.RegTermSize() * TermPairSize::Reg;
		if (term_idx >= term_size && term_idx < term_size + offset.NodeTranslationTermSize())
		{
			type = TermType::NodeTranslation;
			typed_idx = term_idx - term_size;
			nodepair_idx = pair_offset + typed_idx * TermPairSize::NodeTranslation;
			return;
		}
		term_size += offset.NodeTranslationTermSize();
		pair_offset = pair_offset + offset.NodeTranslationTermSize() * TermPairSize::NodeTranslation;
		if (term_idx >= term_size && term_idx < term_size + offset.FeatureTermSize())
		{
			type = TermType::Feature;
			typed_idx = term_idx - term_size;
			nodepair_idx = pair_offset + typed_idx * TermPairSize::Feature;
			return;
		}

		// Not a valid term
		type = TermType::Invalid;
		typed_idx = 0xFFFFFFFF;
		nodepair_idx = 0xFFFFFFFF;
	}
}