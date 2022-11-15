import pickle

def dataset_stats(vectorized_dataset_address: str, coefficient: float):
    """
    Args:
        vectorized_dataset_address: the address of the vectorized pickle file
        coefficient: the coefficient to define popularity

    Returns:
        stats: dict
    """
    with open(vectorized_dataset_address, 'rb') as infile: teamsvecs = pickle.load(infile)
    stats = {}
    teamids, skillvecs, membervecs = teamsvecs['id'], teamsvecs['skill'], teamsvecs['member']

    stats['*nteams'] = teamids.shape[0]
    stats['*nmembers'] = membervecs.shape[1]
    stats['*nskills'] = skillvecs.shape[1]
    row_sums = membervecs.sum(axis=1)
    col_sums = membervecs.sum(axis=0)

    stats['nteams_candidate-idx'] = {k: v for k, v in enumerate(sorted(col_sums.A1.astype(int), reverse=True))}
    stats['*avg_nteams_member'] = col_sums.mean()
    threshold = coefficient * stats['*avg_nteams_member']
    stats['popularity'] = list()

    for index, appearance in zip(teamids, col_sums.getA1()):
        if threshold <= appearance:
            stats['popularity'].append('P')
        else:
            stats['popularity'].append('NP')

    popular_portion = stats['popularity'].count('P') / stats['*nmembers']
    stats['distributions'] = {'P':  popular_portion, 'NP': 1 - popular_portion}
    return stats

dataset_stats('../processed/dblp-toy/teamsvecs.pkl', 1)