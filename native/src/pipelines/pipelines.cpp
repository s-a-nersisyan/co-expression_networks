#include <string>
#include <utility>

#include "../correlations/correlations.h"
#include "pipelines.h"

int ztest_pipeline(
	float *data_ptr,
	int sample_size,
	int *source_ind_ptr,
	int *target_ind_ptr,
	int start_ind,
	int end_ind,
	int index_size,
	int *ref_ind_ptr,
	int *exp_ind_ptr,
	int ref_ind_size,
	int exp_ind_size,
	float *ref_corrs_ptr,
	float *exp_corrs_ptr,
	float *stat_ptr,
	float *pvalue_ptr,
	const std::string correlation,
	const std::string alternative
) {
	if (correlation == SPEARMAN) {
		pearsonr(
			data_ptr,
			sample_size,
			source_ind_ptr,
			target_ind_ptr,
			ref_corrs_ptr,
			start_ind,
			end_ind,
			index_size,
			ref_ind_ptr,
			ref_ind_size
		);
		
		pearsonr(
			data_ptr,
			sample_size,
			source_ind_ptr,
			target_ind_ptr,
			exp_corrs_ptr,
			start_ind,
			end_ind,
			index_size,
			exp_ind_ptr,
			exp_ind_size
		);
	} else {
		pearsonr(
			data_ptr,
			sample_size,
			source_ind_ptr,
			target_ind_ptr,
			ref_corrs_ptr,
			start_ind,
			end_ind,
			index_size,
			ref_ind_ptr,
			ref_ind_size
		);

		pearsonr(
			data_ptr,
			sample_size,
			source_ind_ptr,
			target_ind_ptr,
			exp_corrs_ptr,
			start_ind,
			end_ind,
			index_size,
			exp_ind_ptr,
			exp_ind_size
		);
	}

	ztest_unsized(
		ref_corrs_ptr, ref_ind_size,
		exp_corrs_ptr, exp_ind_size,
		stat_ptr, pvalue_ptr,
		start_ind, end_ind,
		correlation,
		alternative	
	);
	
	return 0;
}
