from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.stats import chisquare, wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt

from data_structures import ResultContainer


"""
    A function for calculating the proportions for POS tags and dependency relations. 
    The calculation for each tag is as follows:

                     (number of words assigned the UPOS tag / dep relation in question in the corpus)  
    tag_proportion = ------------------------------------------------------------------------
                                    ( total number of words in the corpus)

"""


# define a function that calculates the proportions for a given tag and outputs the result to a directory.
# the function also outputs a barchart comparison for all the proportions in both datasets
def export_tag_proportions(first_list, second_list, output_dir, mode, rc: ResultContainer):
    mode = mode.lower()
    if mode not in ["deprel", "upos"]:
        raise Exception("Error, mode should be either 'deprel' or 'upos'")

    if mode == "deprel":
        mode_full = "Dependency relation"
    else:
        mode_full = "Part-of-speech tag"

    print(f"Exporting {mode} proportions")
    with open(f"{output_dir}/{mode}_proportions.tsv", "w", encoding="utf-8") as wf:
        # write the header line
        wf.write("\t".join([f"{mode}_name", f"{rc.dataset_names[0]}_corpus", f"{rc.dataset_names[1]}_corpus", "Chisq",
                            "P-value", "p<0.05?"]) + "\n")

        total_first_words = 0
        total_second_words = 0
        first_tag_counts = Counter()
        second_tag_counts = Counter()
        # get the counts for each tag in the first corpus
        for first_sent in first_list:
            for first_tok in first_sent:
                total_first_words += 1
                first_tag_counts[first_tok[mode]] += 1

        # get the counts for each tag in the second corpus
        for second_sent in second_list:
            for second_tok in second_sent:
                total_second_words += 1
                second_tag_counts[second_tok[mode]] += 1

        # write the proportions and calculate the chisquare test for each tag (only tags that appear in both datasets
        # are considered)
        first_proportions = dict()
        second_proportions = dict()
        for tag in (set(first_tag_counts.keys()) & set(second_tag_counts.keys())):
            n1 = first_tag_counts[tag]
            n2 = second_tag_counts[tag]
            first_proportion = n1 / total_first_words
            first_proportions[tag] = first_proportion
            second_proportion = n2 / total_second_words
            second_proportions[tag] = second_proportion

            f_obs = np.array([n1, n2])
            f_exp = np.array([(((n1 + n2) * total_first_words) / (total_first_words + total_second_words)),
                              (((n1 + n2) * total_second_words) / (total_first_words + total_second_words))])

            chi2, p_value = chisquare(f_obs, f_exp)

            significance = "Yes" if p_value < 0.05 else "No"
            wf.write("\t".join([tag, str(first_proportion), str(second_proportion), str(chi2),
                                str(p_value), significance]) + "\n")

    # draw a barchart
    df1 = pd.DataFrame(list(first_proportions.items()), columns=[mode, 'Proportion'])
    df1['Dataset'] = rc.dataset_names[0]

    df2 = pd.DataFrame(list(second_proportions.items()), columns=[mode, 'Proportion'])
    df2['Dataset'] = rc.dataset_names[1]

    barchart_df = pd.concat([df1, df2])

    difference_df = barchart_df.pivot(index=mode, columns='Dataset', values='Proportion').reset_index()
    difference_df['Difference'] = difference_df[rc.dataset_names[1]] - difference_df[rc.dataset_names[0]]
    sorted_tags = difference_df.sort_values(by='Difference', ascending=False)[mode]
    if mode == "upos":
        rc.pos_top_4 = [",".join([f"{tg}-{abs(difference_df.loc[difference_df[mode] == tg, 'Difference'].values[0]):.2f}" for tg in sorted_tags[:-5:-1]]),
                        ",".join([f"{tg}-{abs(difference_df.loc[difference_df[mode] == tg, 'Difference'].values[0]):.2f}" for tg in sorted_tags[:4]])]
    elif mode == "deprel":
        rc.deprel_top_4 = [",".join([f"{tg}-{abs(difference_df.loc[difference_df[mode] == tg, 'Difference'].values[0]):.2f}" for tg in sorted_tags[:-5:-1]]),
                           ",".join([f"{tg}-{abs(difference_df.loc[difference_df[mode] == tg, 'Difference'].values[0]):.2f}" for tg in sorted_tags[:4]])]

    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x=mode, y='Proportion', hue='Dataset', data=barchart_df, palette='cubehelix',
                edgecolor="0.2", order=sorted_tags)

    # Adding labels and title
    plt.title(f'{mode_full} Proportion Comparison')
    plt.ylabel('Proportion')
    plt.xlabel(mode_full)
    plt.xticks(rotation=45)
    plt.legend(title='Treebank', loc='upper right')

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25)

    # Caption and display the plot
    # TODO: adjust the figure number so that it reflects the number of visualizations actually generated by the current
    #  process
    caption = (r"$\bf{Figure:}$" + f"Barchart showing the proportion of each {mode_full} tag for each treebank. The "
               f"ordering of the tags is determined by the difference between the tag proportions between the two "
               f"treebanks, with the tags on the left end being more typical (i.e. occurring with a higher proportion "
               f"difference) of the second treebank, while the tags on the right end being more typical of the first "
               f"treebank.")
    fig.text(0, 0.01, caption, wrap=True, fontsize=10)

    plt.savefig(f"{output_dir}/{mode}_barcharts.png")