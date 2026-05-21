from scipy.stats import shapiro
from statistics import stdev
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from data_structures import ComparisonConfig, ResultContainer


# utility function to pass in to the function that performs the permutation test
def mean_for_permutation(x, y, axis=0):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


# test whether the data is normally distributed
def check_for_normality(data, sign_level=0.05):
    shapiro_test = shapiro(data)

    return (shapiro_test.pvalue < sign_level, shapiro_test.pvalue, shapiro_test.statistic)


# utility function to indicate the level of statistical significance
def p_to_star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


# function that takes some data and performs a bootstrapping process. It returns a list of bootstrapped statistics 
# that can then be used to inspect the true distribution. 
def return_bootstrapped(data, statistic, n=1000):
    bootstrapped_values = list()
    
    for sample_index in range(n):
        bootstrap_sample = random.choices(data, k=len(data))
        try:
            bootstrapped_values.append(statistic(bootstrap_sample))
        except:
            raise Exception(f"problem with {bootstrap_sample=}")
    
    return bootstrapped_values


# used for getting the ci for cohen's d (difference of means)
def return_bootstrapped_ci(first_data, second_data, statistic, n=1000):
    bootstrapped_values = list()

    for resample_indey in range(n):
        first_resample = random.choices(first_data, k=len(first_data))
        second_resample = random.choices(second_data, k=len(second_data))
        try:
            bootstrapped_values.append(statistic(first_resample, second_resample))
        except:
            raise Exception(f"problem when resampling for bootstrap test!!!")
        
    # return the relevant quantiles for a 95% CI
    return np.percentile(bootstrapped_values, 2.5), np.percentile(bootstrapped_values, 97.5)


# A basic implementation of a permutation test for testing whether two samples come from the same distribution.
# Taken from the following article: https://www.geeksforgeeks.org/machine-learning/permutation-tests-in-machine-learning/
def permutation_test_basic(group_a, group_b, num_permutations=10000):
    observed_statistic = np.mean(group_a) - np.mean(group_b)
    combined_data = np.concatenate((group_a, group_b))
    permuted_statistics = []

    for _ in range(num_permutations):
        np.random.shuffle(combined_data)
        perm_group_a = combined_data[:len(group_a)]
        perm_group_b = combined_data[len(group_a):]
        perm_statistic = np.mean(perm_group_a) - np.mean(perm_group_b)
        permuted_statistics.append(perm_statistic)

    print(np.sum(np.abs(permuted_statistics) >= np.abs(observed_statistic)))
    p_value = np.sum(np.abs(permuted_statistics) >= np.abs(observed_statistic)) / num_permutations
    return p_value


# basic implementation for the Cohen's d effect size calculation.
# Taken from the following article: https://www.askpython.com/python/examples/cohens-d-python
def cohens_d(group1, group2):
    mean1, mean2 = np.mean(group1), np.mean(group2)
     
    # pooled standard deviation
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
     
    # Final calculation
    d = (mean1 - mean2) / pooled_std
     
    return d


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

