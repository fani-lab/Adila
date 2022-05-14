import reranking
def reranking_logic(item_attributes,author_count,length_of_qualified_authors,list_of_author_id,author_id_to_dictionary):
    distribution_list = {"p": 0.5, "np": 0.5}
    rerank_indices = reranking.rerank(
            item_attributes,
            distribution_list,
            max_na=None,
            k_max=None,
            algorithm="det_greedy",
            verbose=False,
        )
    item_attribute_reranked = [item_attributes[i] for i in rerank_indices]
    list_of_author_id = list(author_id_to_dictionary)
    reaarrangedDict = {}
    for k in rerank_indices:
        reaarrangedDict[list_of_author_id[k]] = author_id_to_dictionary[list_of_author_id[k]]
    print(f"reranking list of items: {list(reaarrangedDict.keys())[:author_count]}")
    return item_attribute_reranked,distribution_list