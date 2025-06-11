from collections import defaultdict, Counter
import os
from nltk import ngrams
from statistics import mean, stdev, median
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data_structures import ResultContainer, ComparisonConfig

"""
    Functions to calculate the average relative frequencies of n-grams for various values of n, 
    their corresponding NGDs (N-gram Diversity Scores), and produce a stripplot to visualize the relative
    frequencies. 
"""


# define a function that will calculate the n-gram diversity score.
def export_ngrams(segment_list, output_dir, mode, n_list, rc: ResultContainer, cc: ComparisonConfig, 
                  export_all_ngrams=False, threshold=20):
    first_segm_sen_list = list()
    for segm in segment_list:
        # We have to ignore punctuation and other misc. characters here, so that they do not show up within n-grams.
        first_segm_sen_list.append([[token["form"].lower() for token in sent if token["form"] not in cc.ngrams_ignore] for
                                    sent in segm])
    
    ngd_dict = defaultdict(list)

    if not os.path.isdir(f"{output_dir}/n-grams"):
        os.mkdir(f"{output_dir}/n-grams")

    if mode == "first":
        dataset_name = rc.dataset_names[0]
    elif mode == "second":
        dataset_name = rc.dataset_names[1]

    if not os.path.isdir(f"{output_dir}/n-grams/{dataset_name}"):
        os.mkdir(f"{output_dir}/n-grams/{dataset_name}")

    print(f"Exporting ngrams to {output_dir}/n-grams/{dataset_name}")
    with (open(f"{output_dir}/n-grams/{dataset_name}/n-grams_summary.tsv", "w", encoding="utf-8") as wf_summary):
        wf_summary.write("n\tNGD\n")
        for n in n_list:
            if not os.path.isdir(f"{output_dir}/n-grams/{dataset_name}/{n}"):
                os.mkdir(f"{output_dir}/n-grams/{dataset_name}/{n}")

            i = 1
            for seg in first_segm_sen_list:
                segment_frequencies = Counter()
                for sentence in seg:
                    if len(sentence) >= n:
                        curr_sent_counter = Counter(list(ngrams(sentence, n)))

                        for key, value in curr_sent_counter.items():
                            segment_frequencies[key] += value

                """# convert the frequencies to relative frequencies (freq. per million)
                relative_frequencies = dict()
                corpus_size = sum([len(x) for x in sentence_list])
                for key, value in segment_frequencies.items():
                    relative_frequencies[key] = (value * 1000000) / corpus_size"""

                """# calculate the mean and median relative frequency. Only trees that appear with a minimum frequency per
                # million over some threshold are included in the calculation
                # TODO: move this threshold into the input config object
                mean_rel_freq = mean([x for x in relative_frequencies.values() if x >= threshold])

                median_rel_freq = median([x for x in relative_frequencies.values() if x >= threshold])"""

                # count the total number of n-grams in the segment and then calculate the NGD (N-gram Diversity Score)
                frequencies_sum = sum(segment_frequencies.values())
                ngd = len(segment_frequencies.keys()) / frequencies_sum
                ngd_dict[n].append(ngd)

                if export_all_ngrams:
                    with open(f"{output_dir}/n-grams/{dataset_name}/{n}/segment_{i}.txt", "w", encoding="utf-8") \
                            as wf_ngrams:

                        wf_ngrams.write(f"===========================================\n"
                                        f"Segment {n}-grams\n"
                                        f"===========================================\n"
                                        f"Segment {n}-gram Diversity Score: {ngd}\n"
                                        f"===========================================\n"
                                        f"FORM\tABSOLUTE FREQ\n")

                        for ngram_form in sorted(segment_frequencies, key=lambda x: segment_frequencies[x], reverse=True):
                            wf_ngrams.write("\t".join([str(ngram_form), str(segment_frequencies[ngram_form])]) + "\n")
                        wf_ngrams.write("\n")

                i += 1

            # calculate the mean and standard deviation for this n and store the results
            mean_ngd = mean(ngd_dict[n])
            stdev_ngd = stdev(ngd_dict[n])
            wf_summary.write(f"{n}\t{mean_ngd}\n")
            if mode == "first":
                """rc.mean_rf_first[n] = mean_rel_freq
                rc.median_rf_first[n] = median_rel_freq"""
                rc.mean_ngd_first[n] = mean_ngd
                rc.stdev_ngd_first[n] = stdev_ngd
            elif mode == "second":
                """rc.mean_rf_second[n] = mean_rel_freq
                rc.median_rf_second[n] = median_rel_freq"""
                rc.mean_ngd_second[n] = mean_ngd
                rc.stdev_ngd_second[n] = stdev_ngd

    return ngd_dict


# function to visualize the n-gram frequencies
def draw_stripplot(first_dict, second_dict, output_dir, n_list, rc: ResultContainer):
    print("Drawing n-gram strip plots")
    # Create lists to hold all data
    n_values = list()
    frequencies = list()
    datasets = list()

    for n, freqs in first_dict.items():
        n_values.extend([n] * len(freqs))
        frequencies.extend(freqs)
        datasets.extend([rc.dataset_names[0]] * len(freqs))

    for n, freqs in second_dict.items():
        n_values.extend([n] * len(freqs))
        frequencies.extend(freqs)
        datasets.extend([rc.dataset_names[1]] * len(freqs))

    freqs_df = pd.DataFrame({"n": n_values, "Frequency per million": frequencies, "Dataset": datasets})

    # Create subplots
    # fig, axes = plt.subplots(ceil(len(n_list) / 3), 3, figsize=(15, 5), sharey=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)

    # Iterate over the n values and create a strip plot for each one
    for i, n in enumerate(n_list):
        ax = axes.flatten()[i]  # Select the current axis
        sns.stripplot(data=freqs_df[freqs_df["n"] == n], x="Dataset", y="Frequency per million", hue="Dataset",
                      ax=ax, jitter=0.6, dodge=True, palette="cubehelix", marker=".", size=5,
                      legend=False)

        ax.set_title(f"n = {n}")
        ax.set_xlabel("Treebank")
        if i % 2 == 0:  # Only set y-label for the first plot in the row
            ax.set_ylabel("Frequency per million")

        for dataset, x_pos, color in zip([rc.dataset_names[0], rc.dataset_names[1]], [0, 1], ["green", "red"]):
            # plot the mean frequency per million
            # median_freq = rc.median_rf_first[n] if dataset == rc.dataset_names[0] else rc.median_rf_second[n]
            if dataset == rc.dataset_names[0]:
                mean_freq = rc.mean_rf_first[n]
                coordinates = [0.50, 0.95]
            else:
                mean_freq = rc.mean_rf_second[n]
                coordinates = [0.50, 0.85]
            # ax.axhline(mean_freq, linestyle="dashed", color=color, linewidth=2,
            #           label=f"{dataset} Median")
            ax.text(coordinates[0], coordinates[1], f"{dataset} Mean = {mean_freq:.2f}", color=color, fontsize=14,
                    ha='center', va='top', transform=ax.transAxes)

    fig.suptitle("N-gram Frequency per Million Comparison")
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    caption = (r"$\bf{Figure\ 4:}$" + f"Stripplot showing the frequencies per million for all n-grams and for "
               f"different values of n.")
    fig.text(0, 0.01, caption, wrap=True, fontsize=10)

    plt.savefig(f"{output_dir}/ngram_stripplot.png")


# function for visualizing the n-gram rank-frequency distribution for a specific n
def draw_rankfreq(first_dict, second_dict, output_dir, n, rc: ResultContainer):
    print(f"Plotting the rank-frequency distribution for n={n}")

    # The power-law function to fit the data to
    def zipf(rank, alpha, C):
        return C * rank ** (-alpha)

    # prepare the data
    first_df = pd.DataFrame(data={"Ranks": np.arange(1, 1001),
                                  "Frequencies": np.array(first_dict[n][:1000])})
    second_df = pd.DataFrame(data={"Ranks": np.arange(1, 1001),
                                   "Frequencies": np.array(second_dict[n][:1000])})

    """# Fit the power-law model to the data
    first_params, first_covariance = curve_fit(zipf, first_df["Ranks"], first_df["Frequencies"], maxfev=2000)
    first_alpha, first_c = first_params
    print(f"Estimated slope for the {rc.dataset_names[0]} data (α): {first_alpha:.4f}")
    first_df["zipf"] = first_df["Ranks"].apply(zipf, args=(first_alpha, first_c))

    second_params, second_covariance = curve_fit(zipf, second_df["Ranks"], second_df["Frequencies"], maxfev=2000)
    second_alpha, second_c = second_params
    print(f"Estimated slope for the {rc.dataset_names[1]} data (α): {second_alpha:.4f}")
    second_df["zipf"] = second_df["Ranks"].apply(zipf, args=(second_alpha, second_c))"""

    fig, axes = plt.subplots(1, 1, figsize=(15, 5))

    # linear scale plot
    sns.lineplot(x=first_df["Ranks"], y=first_df["Frequencies"], label=f"{rc.dataset_names[0]} frequencies", ax=axes)
    #sns.lineplot(x=first_df["Ranks"], y=first_df["zipf"], label=f"{rc.dataset_names[0]} fit: alpha={first_alpha:.4f}",
    #             ax=axes[0])
    sns.lineplot(x=second_df["Ranks"], y=second_df["Frequencies"], label=f"{rc.dataset_names[1]} frequencies", 
                 ax=axes)
    #sns.lineplot(x=second_df["Ranks"], y=second_df["zipf"],
    #             label=f"{rc.dataset_names[1]} fit: alpha={second_alpha:.4f}", ax=axes[0])
    #axes.set_title("Rank-Frequency Distribution Plot")
    axes.set_xlabel("Rank")
    axes.set_ylabel("Frequency")

    """# log-log scale plot
    sns.lineplot(x=first_df["Ranks"], y=first_df["Frequencies"], label=f"{rc.dataset_names[0]} frequencies", ax=axes[1])
    sns.lineplot(x=first_df["Ranks"], y=first_df["zipf"], label=f"{rc.dataset_names[0]} fit: alpha={first_alpha:.4f}", 
                 ax=axes[1])
    sns.lineplot(x=second_df["Ranks"], y=second_df["Frequencies"], label=f"{rc.dataset_names[1]} frequencies", 
                 ax=axes[1])
    sns.lineplot(x=second_df["Ranks"], y=second_df["zipf"], 
                 label=f"{rc.dataset_names[1]} fit: alpha={second_alpha:.4f}", ax=axes[1])
    axes[1].set_title("Log-log scale plot")
    axes[1].set_xlabel("Log(Rank)")
    axes[1].set_ylabel("Log(Frequency)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")"""

    plt.suptitle(f"{n}-gram Rank-Frequency Distribution")
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    caption = (r"$\bf{Figure\ 5:}$" + f"Line plot showing the rank-frequency distribution for {n}-grams in both "
               f"treebanks. The horizontal axis shows the ranks for the top 1000 most frequent {n}-grams in ascending "
               f"order, while the vertical axis shows the frequency of the {n}-gram for that corresponding rank.")
    fig.text(0, 0.01, caption, wrap=True, fontsize=10)

    plt.savefig(f"{output_dir}/{n}-gram_rank-frequency.png")