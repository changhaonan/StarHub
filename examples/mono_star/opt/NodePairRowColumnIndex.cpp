#include "pcg_solver/solver_configs.h"
#include "star/types/term_offset_types.h"
#include "star/warp_solver/utils/solver_encode.h"
#include "star/warp_solver/NodePair2TermsIndex.h"

void star::NodePair2TermsIndex::blockRowOffsetSanityCheck() {
	// FIXME: What will happen if there are node has no neighbor?? (i.e. node-0, node-1)
	LOG(INFO) << "Sanity check for blocked row offset";

	//Checking of the offset
	std::vector<unsigned> compacted_key, row_offset;
	GArrayView<unsigned> compacted_nodepair(m_symmetric_kv_sorter.valid_sorted_key);
	compacted_nodepair.Download(compacted_key);
	row_offset.clear();
	row_offset.push_back(0);
	for (int i = 1; i < compacted_key.size(); i++) {
		int this_row = encoded_row(compacted_key[i]);
		int prev_row = encoded_row(compacted_key[i - 1]);
		if (this_row != prev_row) {
			row_offset.push_back(i);
		}
	}
	row_offset.push_back(compacted_key.size());

	//Download the gpu offset
	std::vector<unsigned> row_offset_gpu;
	m_blkrow_offset_array.View().Download(row_offset_gpu);
	STAR_CHECK(row_offset.size() == row_offset_gpu.size());
	for (int i = 0; i < row_offset.size(); i++) {
		STAR_CHECK(row_offset[i] == row_offset_gpu[i]);
	}

	LOG(INFO) << "Check done";
}

void star::NodePair2TermsIndex::blockRowLengthSanityCheck() {
	LOG(INFO) << "Sanity check for blocked row length";

	std::vector<unsigned> row_offset;
	m_blkrow_offset_array.View().Download(row_offset);
	STAR_CHECK(row_offset.size() == m_num_nodes + 1);

	//Compute the row size
	std::vector<unsigned> row_length;
	row_length.resize(m_num_nodes);
	for (auto i = 0; i < m_num_nodes; i++) {
		row_length[i] = row_offset[i + 1] - row_offset[i] + 1;
	}

	//Download and check the offset
	std::vector<unsigned> row_length_gpu;
	m_blkrow_length_array.View().Download(row_length_gpu);
	STAR_CHECK_EQ(row_length.size(), row_length_gpu.size());
	for (auto i = 0; i < row_length.size(); i++) {
		STAR_CHECK(row_length[i] == row_length_gpu[i]);
	}

	LOG(INFO) << "Check done";
}

void star::NodePair2TermsIndex::binLengthNonzerosSanityCheck() {
	LOG(INFO) << "Sanity check for bin length and non-zeros";

	std::vector<unsigned> bin_length;
	std::vector<unsigned> blk_length;
	m_blkrow_length_array.View().Download(blk_length);
	unsigned num_bins = divUp(d_node_variable_dim * m_num_nodes, d_bin_size);
	bin_length.resize(num_bins);
	for (unsigned i = 0; i < num_bins; i++) {
		bin_length[i] = 0;
		for (unsigned row_idx = i * d_bin_size; row_idx < (i + 1) * d_bin_size; row_idx++) {
			unsigned blk_row = row_idx / d_node_variable_dim;
			if (blk_row < blk_length.size())
				bin_length[i] = std::max<unsigned>(blk_length[blk_row], bin_length[i]);
		}
		bin_length[i] *= d_node_variable_dim;
	}

	//Download the gpu version for test
	std::vector<unsigned> bin_length_gpu;
	m_binlength_array.View().Download(bin_length_gpu);
	assert(bin_length.size() == bin_length_gpu.size());
	for (size_t i = 0; i < bin_length.size(); i++) {
		assert(bin_length[i] == bin_length_gpu[i]);
	}

	//Next check the non-zero values
	std::vector<unsigned> non_zeros;
	non_zeros.resize(bin_length.size() + 1);
	unsigned sum = 0;
	for (size_t i = 0; i < non_zeros.size(); i++) {
		non_zeros[i] = sum;
		if (i < bin_length.size())
			sum += d_bin_size * bin_length[i];
	}

	std::vector<unsigned> non_zeros_gpu;
	GArrayView<unsigned>(m_binnonzeros_prefixsum.valid_prefixsum_array).Download(non_zeros_gpu);
	assert(non_zeros.size() == non_zeros_gpu.size());
	for (size_t i = 0; i < non_zeros.size(); i++) {
		assert(non_zeros[i] == non_zeros_gpu[i]);
		if (non_zeros[i] != non_zeros_gpu[i]) {
			std::cout << non_zeros[i] << " " << non_zeros_gpu[i] << std::endl;
		}
	}

	LOG(INFO) << "Check done";
}

void star::NodePair2TermsIndex::binBlockCSRRowPtrSanityCheck() {
	LOG(INFO) << "Check of the rowptr for bin-blocked csr";

	//Download the data
	std::vector<unsigned> non_zeros;
	GArrayView<unsigned>(m_binnonzeros_prefixsum.valid_prefixsum_array).Download(non_zeros);

	//Check the row-pointer of JtJ
	std::vector<int> JtJ_rowptr_host; JtJ_rowptr_host.clear();
	for (int i = 0; i < non_zeros.size(); i++) {
		int offset = non_zeros[i];
		for (int j = 0; j < d_bin_size; j++, offset++) {
			JtJ_rowptr_host.push_back(offset);
		}
	}
	std::vector<int> JtJ_rowptr_gpu;
	m_binblocked_csr_rowptr.View().Download(JtJ_rowptr_gpu);
	assert(JtJ_rowptr_gpu.size() == JtJ_rowptr_host.size());
	for (int i = 0; i < JtJ_rowptr_host.size(); i++) {
		assert(JtJ_rowptr_gpu[i] == JtJ_rowptr_host[i]);
	}

	// Log things here
	//for (auto i = 0; i < 100; ++i) {
	//	std::cout << "Generate row_csr " << i << ": " << JtJ_rowptr_host[i] << std::endl;
	//}

	LOG(INFO) << "Check done";
}

void star::NodePair2TermsIndex::binBlockCSRColumnPtrSanityCheck() {
	LOG(INFO) << "Sanity check for the colptr of bin-blocked csr format";

	//Prepare the data
	std::vector<int> JtJ_rowptr, JtJ_column_host;
	std::vector<unsigned> blkrow_offset, Iij_compacted_key;
	m_binblocked_csr_rowptr.View().Download(JtJ_rowptr);
	m_blkrow_offset_array.View().Download(blkrow_offset);
	m_symmetric_kv_sorter.valid_sorted_key.download(Iij_compacted_key);

	//Zero out the elements
	GArrayView<int> JtJ_column_index(m_binblocked_csr_colptr.Ptr(), m_binblocked_csr_colptr.BufferSize());
	JtJ_column_host.resize(JtJ_column_index.Size());
	for (int i = 0; i < JtJ_column_host.size(); i++) {
		JtJ_column_host[i] = 0;
	}

	for (int row_idx = 0; row_idx < d_node_variable_dim * m_num_nodes; row_idx++) {
		int data_offset = JtJ_rowptr[row_idx];
		//First fill the diagonal block
		int blkrow_idx = row_idx / d_node_variable_dim;
		int binwidth_offset = row_idx & 31;
		int column_idx_offset = (data_offset - binwidth_offset) / d_node_variable_dim + binwidth_offset;
		JtJ_column_host[column_idx_offset] = d_node_variable_dim * blkrow_idx;
		for (int i = 0; i < d_node_variable_dim; i++) {
			//JtJ_data_host[data_offset] = diag_values[d_bin_size * blkrow_idx + inblk_offset + d_node_variable_dim * i];
			data_offset += d_bin_size;
		}

		//Then fill the non-block values
		column_idx_offset += d_bin_size;
		int key_begin = blkrow_offset[blkrow_idx];
		int key_end = blkrow_offset[blkrow_idx + 1];
		for (int key_iter = key_begin; key_iter < key_end; key_iter++) {
			auto Iij_key = Iij_compacted_key[key_iter];
			int Iij_column = encoded_col(Iij_key);
			JtJ_column_host[column_idx_offset] = d_node_variable_dim * Iij_column;
			column_idx_offset += d_bin_size;
			for (int i = 0; i < d_node_variable_dim; i++) {
				//JtJ_data_host[data_offset] = nondiag_values[d_bin_size * key_iter + inblk_offset + d_node_variable_dim * i];
				data_offset += d_bin_size;
			}
		}
	}

	//Check the value of column index
	std::vector<int> JtJ_column_gpu;
	JtJ_column_index.Download(JtJ_column_gpu);
	for (int i = 0; i > JtJ_column_host.size(); i++) {
		STAR_CHECK(JtJ_column_host[i] == JtJ_column_gpu[i]);
	}

	LOG(INFO) << "Check done";
}