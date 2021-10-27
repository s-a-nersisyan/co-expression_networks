const float QMEDIAN = 0.5;
const std::string MEDIAN = "median";
const std::string MEAN = "mean";


float mean(
    float *data_ptr,
    int *index_ptr,
    int start_ind,
    int end_ind
);

int _mean(
    float *data_ptr,
    int *starts_ind_ptr,
    int *ends_ind_ptr,
    float *scores_ptr,
    int start_ind,
    int end_ind
);

int __mean(
    float *data_ptr,
    int sources_size,
    float *scores_ptr,
    int start_ind,
    int end_ind
);

float quantile(
    float *data_ptr,
    int *index_ptr,
    int start_ind,
    int end_ind,
    float q=QMEDIAN
);

int _quantile(
    float *data_ptr,
    int *starts_ind_ptr,
    int *ends_ind_ptr,
    float *scores_ptr,
    int start_ind,
    int end_ind,
    float q=QMEDIAN
);

int __quantile(
    float *data_ptr,
    int sources_size,
    float *scores_ptr,
    int start_ind,
    int end_ind,
    float q=QMEDIAN
);
