import reranking
def experimental_metrics(item_attributes,item_attribute_reranked,distribution_list):
    before = reranking.ndkl(item_attributes,distribution_list)
    after = reranking.ndkl(item_attribute_reranked, distribution_list)
    before_ndcg = reranking.ndcg_diff(item_attributes)
    after_ndcg = reranking.ndcg_diff(item_attribute_reranked)
    print(f"NDKL before reranking: {before:.3f}, NDKL after reranking: {after:.3f}")
    print(f"NDCG before reranking {before_ndcg:.3f}, NDCG after reranking  {after_ndcg:.3f}")
    print(f"infeasible metric: {reranking.infeasible(item_attributes,distribution_list)}")
    print(f"reranked infeasible metric : {reranking.infeasible(item_attribute_reranked, distribution_list)}")