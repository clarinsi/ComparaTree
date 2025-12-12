from scipy.stats import shapiro
from statistics import stdev
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from data_structures import ComparisonConfig, ResultContainer


def check_for_normality(data, sign_level=0.05):
    shapiro_test = shapiro(data)

    return (shapiro_test.pvalue < sign_level, shapiro_test.pvalue, shapiro_test.statistic)


# function that takes some data and performs a bootstrapping process. It returns a list of bootstrapped statistics 
# that can then be used to inspect the true distribution. 
def return_bootstrapped(data, statistic, n=10000):
    bootstrapped_values = list()
    
    for sample_index in range(n):
        bootstrap_sample = random.choices(data, k=len(data))
        try:
            bootstrapped_values.append(statistic(bootstrap_sample))
        except:
            raise Exception(f"promblem with {bootstrap_sample=}")
    
    return bootstrapped_values


# An implementation of a permutation test for testing whether two samples come from the same distribution.
# Taken from the following article: https://www.geeksforgeeks.org/machine-learning/permutation-tests-in-machine-learning/
def permutation_test(group_a, group_b, num_permutations=10000):
    observed_statistic = np.mean(group_a) - np.mean(group_b)
    combined_data = np.concatenate((group_a, group_b))
    permuted_statistics = []

    for _ in range(num_permutations):
        np.random.shuffle(combined_data)
        perm_group_a = combined_data[:len(group_a)]
        perm_group_b = combined_data[len(group_a):]
        perm_statistic = np.mean(perm_group_a) - np.mean(perm_group_b)
        permuted_statistics.append(perm_statistic)

    p_value = np.sum(np.abs(permuted_statistics) >= np.abs(observed_statistic)) / num_permutations
    return p_value


# function for visualizing rank-frequency distributions. Takes two lists of raw frequencies, one for each dataset. 
# The lists passed into the function do not necessarily need to be sorted already. The sorting happens function-internally. 
def draw_rankfreq(values, output_file, mode, rc: ResultContainer):
    # prepare the data
    values_df = pd.DataFrame(data={"Ranks": np.arange(1, len(values) + 1),
                                  "Frequencies": np.array(sorted(values, reverse=True))})

    fig, axes = plt.subplots(1, 1, figsize=(15, 5))

    # linear scale plot
    sns.lineplot(x=values_df["Ranks"], y=values_df["Frequencies"], label=f"{rc.dataset_names[0]} frequencies", ax=axes)
    #axes.set_title(f"{mode} Rank-Frequency Distribution Plot")

    # log-log scale plot
    axes.set_title(f"{mode} Rank-Frequency Distribution Plot - Log-Log Scale")
    axes.set_xlabel("Log(Rank)")
    axes.set_ylabel("Log(Frequency)")
    axes.set_xscale("log")
    axes.set_yscale("log")

    plt.suptitle(f"{mode} Rank-Frequency Distribution")
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    caption = (r"$\bf{Figure\ 5:}$" + f"Line plot showing the rank-frequency distribution for {mode}. "
               f"The horizontal axis shows the ranks, while the vertical axis shows the frequency "
               f"for that corresponding rank.")
    fig.text(0, 0.01, caption, wrap=True, fontsize=10)

    plt.savefig(output_file)
    plt.close()


# function that draws rank frequency distributions for a segment in the output directory
def export_rankfreq_plots(freqs, seg_id, mode, output_dir, rc: ResultContainer):
    if not os.path.isdir(os.path.join(output_dir, f"{mode}_rankfreq_plots")):
        os.mkdir(os.path.join(output_dir, f"{mode}_rankfreq_plots"))
    
    draw_rankfreq(freqs, os.path.join(output_dir, f"{mode}_rankfreq_plots", f"seg_{seg_id}_rankfreq_plot.png"), mode, rc)

