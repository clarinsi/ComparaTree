from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.stats import chisquare, wilcoxon, chi2_contingency
from scipy.stats.contingency import margins
import seaborn as sns
import matplotlib.pyplot as plt
import logging

from data_structures import ResultContainer, ComparisonConfig
from stat_utils import p_to_star


"""
    A function for calculating the proportions for POS tags and dependency relations. 
    The calculation for each tag is as follows:

                     (number of words assigned the UPOS tag / dep relation in question in the corpus)  
    tag_proportion = ------------------------------------------------------------------------
                                    ( total number of words in the corpus)

"""


# define a function that calculates the proportions for a given tag and outputs the result to a directory.
# the function also outputs a barchart comparison for all the proportions in both datasets
def export_tag_proportions(first_list, second_list, cc: ComparisonConfig, mode, rc: ResultContainer):
    output_dir = cc.output_dir
    significance_tests = cc.significance_tests
    effect_sizes =cc.effect_sizes
    chisquare_residuals = cc.chisquare_residuals

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
        p_values_dict = dict()
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
            p_values_dict[tag] = p_value

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

    # Add the chi-squared test p-values
    difference_df['p_value'] = difference_df[mode].map(p_values_dict)

    sorted_tags = difference_df.sort_values(by='Difference', ascending=False)[mode]
    if mode == "upos":
        rc.pos_top_4 = [",".join([f"{tg}-{abs(difference_df.loc[difference_df[mode] == tg, 'Difference'].values[0]):.2f}" for tg in sorted_tags[:-5:-1]]),
                        ",".join([f"{tg}-{abs(difference_df.loc[difference_df[mode] == tg, 'Difference'].values[0]):.2f}" for tg in sorted_tags[:4]])]
    elif mode == "deprel":
        rc.deprel_top_4 = [",".join([f"{tg}-{abs(difference_df.loc[difference_df[mode] == tg, 'Difference'].values[0]):.2f}" for tg in sorted_tags[:-5:-1]]),
                           ",".join([f"{tg}-{abs(difference_df.loc[difference_df[mode] == tg, 'Difference'].values[0]):.2f}" for tg in sorted_tags[:4]])]

    # manually set matplotlib's logging level due to some strange debug messages
    plt.set_loglevel("warning")
    logging.getLogger('PIL').setLevel(logging.WARNING)

    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x=mode, y='Proportion', hue='Dataset', data=barchart_df, palette='cubehelix',
                edgecolor="0.2", order=sorted_tags)

    # Add labels and title
    plt.title(f'{mode_full} Proportion Comparison')
    plt.ylabel('Proportion')
    plt.xlabel(mode_full)
    plt.xticks(rotation=45)
    plt.legend(title='Treebank', loc='upper right')

    # Add significance test labels
    significance_text = ""
    if significance_tests:
        
        max_heights = barchart_df.groupby(mode)['Proportion'].max().reindex(sorted_tags)

        for i, tag in enumerate(sorted_tags):
            p = difference_df.loc[difference_df[mode] == tag, 'p_value'].values[0]
            y = max_heights[tag]

            plt.text(i, y + 0.01, p_to_star(p), ha='center', va='bottom', fontsize=9)
        
        significance_text = " The stars above the bars represent whether the difference between the two datasets is statistically significant, " \
                            "where * indicates p < 0.05, ** indicates p < 0.01, and *** indicates p < 0.001."
        
    # Add effect size labels
    effect_text = ""
    if effect_sizes:
        
        max_heights = barchart_df.groupby(mode)['Proportion'].max().reindex(sorted_tags)

        for i, tag in enumerate(sorted_tags):
            d = difference_df.loc[difference_df[mode] == tag, 'Difference'].values[0]
            y = max_heights[tag]

            plt.text(i, y + 0.02, f"dp = {d:.3f}", ha='center', va='bottom', fontsize=9)
        
        effect_text = " The 'dp' value above the bars indicates the difference in proportion between the two datasets for each tag."

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25)

    # Caption and display the plot
    # TODO: adjust the figure number so that it reflects the number of visualizations actually generated by the current
    #  process
    caption = (r"$\bf{Figure:}$" + f"Barchart showing the proportion of each {mode_full} tag for each treebank. The "
               f"ordering of the tags is determined by the difference between the tag proportions between the two "
               f"treebanks, with the tags on the left end being more typical (i.e. occurring with a higher proportion "
               f"difference) of the second treebank, while the tags on the right end being more typical of the first "
               f"treebank.{significance_text}{effect_text}")
    fig.text(0, 0.01, caption, wrap=True, fontsize=10)

    plt.savefig(f"{output_dir}/{mode}_barcharts.png")

    if chisquare_residuals:
        #draw_residuals_scatterplot(first_tag_counts, second_tag_counts, mode_full, cc, rc)
        draw_residuals_dotplot(first_tag_counts, second_tag_counts, mode_full, cc, rc)
        #draw_residuals_barchart(first_tag_counts, second_tag_counts, mode_full, cc, rc)


# Function to calculate chisquare residuals and draw them on a combined scatterplot for both compared treebanks
def draw_residuals_scatterplot(first_counts, second_counts, mode, cc:ComparisonConfig, rc: ResultContainer):
    output_dir = cc.output_dir
    first_name = rc.dataset_names[0]
    second_name = rc.dataset_names[1]

    combined_df = pd.DataFrame([first_counts, second_counts], index=[first_name, second_name])
    combined_df = combined_df.fillna(0).astype(int)

    # chi-squared test
    _, chi2_p, _, expected = chi2_contingency(combined_df)

    # standardized residuals
    observed = combined_df.values
    n = observed.sum()
    rsum, csum = margins(observed)
    rsum = rsum.astype(np.float64)
    csum = csum.astype(np.float64)
    v = csum * rsum * (n - rsum) * (n - csum) / n**3

    std_residuals_matrix = (observed - expected) / np.sqrt(v)
    residuals_df = pd.DataFrame(std_residuals_matrix, index=combined_df.index, columns=combined_df.columns)

    print(residuals_df)

    # Plot the scatterplot
    plot_df = residuals_df.T
    plot_df.columns = ['Residual_A', 'Residual_B']
    plot_df["Tag"] = plot_df.index

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))

    scatter = sns.scatterplot(
        data=plot_df, 
        x="Residual_B", 
        y="Residual_A", 
        hue="Residual_A", 
        palette="GnBu_r", 
        s=150,
        edgecolor="black",
        legend=False
    )

    # Add POS tag labels to the points
    for i in range(plot_df.shape[0]):
        plt.text(
            plot_df.Residual_B.iloc[i] + 0.1, 
            plot_df.Residual_A.iloc[i] + 0.1, 
            plot_df.Tag.iloc[i], 
            fontsize=11, 
            fontweight="bold",
            alpha=0.7
        )

    # Add additional horizontal and vertical lines
    plt.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    plt.axvline(0, color="gray", linestyle="-", linewidth=0.5)

    # Axis labels
    plt.title(f"{mode} Standardized Residuals", fontsize=16, pad=20)
    plt.xlabel(f"Residual in {second_name}", fontsize=12)
    plt.ylabel(f"Residual in {first_name}", fontsize=12)

    # Set equal limits to maintain the diagonal symmetry
    max_val = max(abs(plot_df["Residual_A"].max()), abs(plot_df["Residual_A"].min())) + 1
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)

    p_text = f"{chi2_p:.4e}" if chi2_p < 0.001 else f"{chi2_p:.4f}"
    caption = f"Figure 1: Standardized residuals for {mode}. Chi-square test p-value = {p_text}"
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12, style='italic', 
                color='dimgray')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{mode}_residuals_scatter.png")

    sns.set_theme(style="white")


# Function to calculate chisquare residuals and draw them for a single corpus
def draw_residuals_dotplot(first_counts, second_counts, mode, cc:ComparisonConfig, rc: ResultContainer):
    output_dir = cc.output_dir
    first_name = rc.dataset_names[0]
    second_name = rc.dataset_names[1]
    font_size = 28

    combined_df = pd.DataFrame([first_counts, second_counts], index=[first_name, second_name])
    combined_df = combined_df.fillna(0).astype(int)

    # chi-squared test
    _, chi2_p, _, expected = chi2_contingency(combined_df)

    # standardized residuals
    observed = combined_df.values
    n = observed.sum()
    rsum, csum = margins(observed)
    rsum = rsum.astype(np.float64)
    csum = csum.astype(np.float64)
    v = csum * rsum * (n - rsum) * (n - csum) / n**3

    std_residuals_matrix = (observed - expected) / np.sqrt(v)
    residuals_df = pd.DataFrame(std_residuals_matrix, index=combined_df.index, columns=combined_df.columns)

    # Export the residuals dataframe and plot the dotplot only for the 5 highest and lowest residuals 
    # (using only Corpus B residuals atm)
    plot_df = residuals_df.iloc[1].sort_values(ascending=False).reset_index()
    plot_df.columns = ["Tag", "Residual"]

    plot_df.to_csv(f"{output_dir}/{mode}_residuals_df.tsv", sep="\t")

    # drop rows that are not to be plotted
    top_5 = plot_df.nlargest(5, "Residual") 
    bottom_5 = plot_df.nsmallest(5, "Residual")

    plot_df = pd.concat([top_5, bottom_5]).drop_duplicates().sort_values("Residual", ascending=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 11))
    
    scatter = sns.scatterplot(
        data=plot_df, 
        x='Residual', 
        y='Tag', 
        hue='Residual', 
        palette='crest_r', 
        s=400,  
        edgecolor='black', 
        zorder=2,
        legend=False
    )

    # Draw the tail (horizontal line)
    # color=colors[i] matches the tail to the dot
    colors = sns.color_palette("crest", n_colors=len(plot_df), )

    for i, (idx, row) in enumerate(plot_df.iterrows()):
        plt.hlines(y=row['Tag'], xmin=0, xmax=row['Residual'], 
                   color=colors[i], linewidth=5, alpha=0.6)

    # the zero vertical line
    plt.axvline(0, color='black', linewidth=3, zorder=3)

    plt.title(f'{second_name}\n{mode}\nStandardized Chi-squared Residuals', fontsize=font_size)
    plt.xlabel(f'<-- Lower than expected || Higher than expected -->', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=24)
    scatter.set(ylabel=None)

    # Ensure the X-axis is symmetrical
    #limit = max(abs(plot_df['Residual'].min()), abs(plot_df['Residual'].max())) + 1
    #plt.xlim(-limit, limit)

    #plt.tight_layout()

    print(f"Chi-squared test for {mode}: p-value = {chi2_p}")

    #caption = f"Figure 1: Standardized chi-squared residuals for {mode}. Chi-squared test p-value = {chi2_p}"
    #plt.figtext(0.5, 0.02, caption, wrap=True, horizontalalignment='center', fontsize=12, style='italic', 
    #            color='dimgray')
    
    plt.savefig(f"{output_dir}/{mode}_residuals_dotplot.png")

    sns.set_theme(style="white")


# Function to calculate chisquared test residuals and draw a barchart for the positive labels for each corpus
def draw_residuals_barchart(first_counts, second_counts, mode, cc:ComparisonConfig, rc: ResultContainer):
    def draw_chart(data, mode, chi2_p, filepath):
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.barplot(x=data.values, y=data.index, palette="mako")

        #ax.tick_params(axis='x', labelrotation=45)

        plt.title(f'{mode} Standardized Chi-squared Residuals')
        plt.ylabel(f"{mode}")
        plt.xlabel(f"Residual")

        #caption = f"Figure 1: Standardized chi-squared residuals for {mode}. Chi-squared test p-value = {chi2_p}"
        #plt.figtext(0.5, 0.02, caption, wrap=True, horizontalalignment='center', style='italic', color='dimgray')

        plt.tight_layout()
        plt.savefig(filepath)
        plt.clf()


    output_dir = cc.output_dir
    first_name = rc.dataset_names[0]
    second_name = rc.dataset_names[1]

    combined_df = pd.DataFrame([first_counts, second_counts], index=[first_name, second_name])
    combined_df = combined_df.fillna(0).astype(int)

    # chi-squared test
    _, chi2_p, _, expected = chi2_contingency(combined_df)

    # standardized residuals
    observed = combined_df.values
    n = observed.sum()
    rsum, csum = margins(observed)
    rsum = rsum.astype(np.float64)
    csum = csum.astype(np.float64)
    v = csum * rsum * (n - rsum) * (n - csum) / n**3

    std_residuals_matrix = (observed - expected) / np.sqrt(v)
    residuals_df = pd.DataFrame(std_residuals_matrix, index=combined_df.index, columns=combined_df.columns)

    print(residuals_df)

    corpus1_df = residuals_df.loc[first_name].sort_values(ascending=False)

    print(corpus1_df)

    corpus1_df = corpus1_df[corpus1_df>0]

    corpus2_df = residuals_df.loc[second_name].sort_values(ascending=False)
    corpus2_df = corpus2_df[corpus2_df>0]

    # draw the barcharts
    draw_chart(corpus1_df, mode, chi2_p, f"{output_dir}/{first_name}_{mode}_residuals_barchart.png")
    draw_chart(corpus2_df, mode, chi2_p, f"{output_dir}/{second_name}_{mode}_residuals_barchart.png")
