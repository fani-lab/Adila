from collections import Counter
import matplotlib.pyplot as plt
def plot_stats(author_id):
    x = Counter(author_id)
    x_reverse = dict(sorted(x.items(), key=lambda item: item[1], reverse=True))
    plt.plot(list(Counter(x_reverse).values()))
    plt.xlabel("Author")
    plt.ylabel("Number of papers published")
    plt.title("Popularity Bias of paper publication by author")
    plt.savefig('./output/graph',dpi=300)