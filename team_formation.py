from collections import Counter

def convert_author_id_to_attributes(authorList,author_id):
    print(f'suggested author list: {list(set(authorList))[:4]}')
    author_id_to_dictionary = dict()
    author_id_count = dict(Counter(author_id))
    for i in authorList:
        if i in author_id_count:
            if author_id_count[i] > 3:
                author_id_to_dictionary[i] = "p"
            else:
                author_id_to_dictionary[i] = "np"
    return author_id_to_dictionary

def form_teams_with_skills(list_of_skills, author_Instance,author_id):

    author_list =list()
    for list_item in list_of_skills:
        for author in author_Instance:
            if list_item in author.get_skills():
                author_list.append(author.id)
    print(f'suggested author list: {list(set(author_list))[:4]}')
    author_id_to_dictionary = convert_author_id_to_attributes(author_list,author_id)
    return author_list,author_id_to_dictionary