settings = {
    'fair': {
        'metrics': {'ndkl', 'skew', 'exp', 'expu'},
        'k_max': 100, #cutoff for the reranking algorithms
        'alpha': 0.1, #the significance value for fa-ir algortihm
        'np_ratio': None, #nonpopularity ratio, if None, based on distribution in dataset from popularity_threshold; default: None; Eg. 0.5'
        'popularity_thresholding': 'avg', #we determine whether an expert is popular or otherwise based on avg teams per experts of equal auc
    },
    'utility_metrics': {'ndcg_cut_2,5,10,20,50,100', 'map_cut_2,5,10,20,50,100'}, # any pytrec_eval metrics
    'parallel': False,
    'core': -1, #number of cores to dedicate to parallel run, -1 means all available cores
}