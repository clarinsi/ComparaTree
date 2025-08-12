import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from statistics import mean, stdev, median
from math import log, sqrt, ceil

from data_structures import ResultContainer


"""
    Syntactic complexity calculations for each sentence in a corpus 
    using the mean dependency distance (MDD) and the normalized dependency distance of a sentence (NDD)
"""


# mean dependency distance function - a precursor to the normalized dependency distance function
def mdd(sentence):
    list_of_dd = list()
    for tok in sentence:
        # skip multi-word tokens
        if not isinstance(tok["id"], int):
            continue
         
        if tok["deprel"] not in ["punct", "root"]:
            list_of_dd.append(abs(int(tok["id"]) - int(tok["head"])))

    return sum(list_of_dd) / len(list_of_dd) if len(list_of_dd) > 0 else np.nan


# normalized dependency distance function
def ndd(sentence, sent_mdd):
    if sent_mdd is np.nan:
        return np.nan

    root_distance = 0
    sentence_length = 0
    for tok in sentence:
        if tok["deprel"] == "root":
            root_distance = int(tok["id"])

        if tok["deprel"] not in ["punct", "root"]:
            sentence_length += 1

    if root_distance < 1 or sentence_length < 1:
        print("WARNING: Can't get root or sentence length for one of the sentences")
        return np.nan

    return abs(log(sent_mdd / sqrt(root_distance * sentence_length)))


# main function to make the syntactic complexity calculation
def export_syntactic_complexity_measure(first_dataset: list, second_dataset: list, output_dir, rc: ResultContainer):
    def draw_sc_histogram(first_data, second_data, first_mean, second_mean, mode):
        mode = mode.lower()
        if mode == "mdd":
            mode_full = "Mean Dependency Distance"
        else:
            mode_full = "Normalized Dependency Distance"

        # draw histogram
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        sns.histplot(first_data, kde=False, ax=axes[0])
        axes[0].set_title(f"Treebank={rc.dataset_names[0]}")
        axes[0].set(xlabel=f"Range of values for {mode_full}")
        axes[0].axvline(x=first_mean, color='red', linestyle='-', linewidth=2)
        axes[0].text(0.95, 0.95, f"Mean = {first_mean:.3f}", color='red', fontsize=14, ha='right', va='top',
                     transform=axes[0].transAxes)
        axes[0].text(0.95, 0.80, f"Standard deviation = {stdev([x for x in first_data if x is not np.nan]):.3f}",
                     color='orange', fontsize=14, ha='right', va='top', transform=axes[0].transAxes)

        sns.histplot(second_data, kde=False, ax=axes[1])
        axes[1].set_title(f"Treebank={rc.dataset_names[1]}")
        axes[1].set(xlabel=f"Range of values for {mode_full}")
        axes[1].axvline(x=second_mean, color='red', linestyle='-', linewidth=2)
        axes[1].text(0.95, 0.95, f"Mean = {second_mean:.3f}", color='red', fontsize=14, ha='right', va='top',
                     transform=axes[1].transAxes)
        axes[1].text(0.95, 0.80, f"Standard deviation = {stdev([x for x in second_data if x is not np.nan]):.3f}",
                     color='orange', fontsize=14, ha='right', va='top', transform=axes[1].transAxes)

        fig.suptitle(f"{mode_full} Histogram")

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.14)
        caption = (r"$\bf{Figure:}$" + f"Histogram showing the frequency distribution for {mode_full} in both "
                   f"treebanks. The blue bars represent the number of observations for each value of the measure. The "
                   f"red vertical line represents the mean of the {mode_full}.")
        fig.text(0, 0.01, caption, wrap=True, fontsize=10)

        plt.savefig(f"{output_dir}/syntactic_complexity_{mode}_histogram.png")

    print(f"Calculating syntactic complexity (MDD and NDD)")
    with open(f"{output_dir}/syntactic_complexity.tsv", "w", encoding="utf-8") as wf_sc:
        sc_measurements = defaultdict(list)
        for dataset in [first_dataset, second_dataset]:
            # TODO: This should output dataset names as given in the config
            # introduce dataset and write header
            if dataset == first_dataset:
                wf_sc.write(f"=============================================\n"
                            f"{rc.dataset_names[0]} SYNTACTIC COMPLEXITY MEASURES:\n"
                            f"=============================================\n\n")
                wf_sc.write("\t".join(["SENT_ID", "MDD", "NDD"]) + "\n")
            if dataset == second_dataset:
                wf_sc.write(f"\n=============================================\n"
                            f"{rc.dataset_names[1]} SYNTACTIC COMPLEXITY MEASURES:\n"
                            f"=============================================\n\n")
                wf_sc.write("\t".join(["SENT_ID", "MDD", "NDD"]) + "\n")

            # calculate sc measures
            i = 0
            for sent in dataset:
                i += 1
                sent_mdd = mdd(sent)
                sent_ndd = ndd(sent, sent_mdd)
                if dataset == first_dataset:
                    sc_measurements["first_mdd"].append(sent_mdd)
                    sc_measurements["first_ndd"].append(sent_ndd)
                else:
                    sc_measurements["second_mdd"].append(sent_mdd)
                    sc_measurements["second_ndd"].append(sent_ndd)

                wf_sc.write("\t".join([str(i), str(sent_mdd), str(sent_ndd)]) + "\n")

            # calculate the mean and stdev
            if dataset == first_dataset:
                mean_mdd = mean([x for x in sc_measurements["first_mdd"] if x is not np.nan])
                mean_ndd = mean([x for x in sc_measurements["first_ndd"] if x is not np.nan])
                wf_sc.write("\t".join(["MEAN", str(mean_mdd), str(mean_ndd)]) + "\n")
                rc.mean_mdd[0] = mean_mdd
                rc.mean_ndd[0] = mean_ndd

                stdev_mdd = stdev([x for x in sc_measurements["first_mdd"] if x is not np.nan])
                stdev_ndd = stdev([x for x in sc_measurements["first_ndd"] if x is not np.nan])
                wf_sc.write("\t".join(["STANDARD DEVIATION", str(stdev_mdd), str(stdev_ndd)]) + "\n")
                rc.stdev_mdd[0] = stdev_mdd
                rc.stdev_ndd[0] = stdev_ndd

            else:
                mean_mdd = mean([x for x in sc_measurements["second_mdd"] if x is not np.nan])
                mean_ndd = mean([x for x in sc_measurements["second_ndd"] if x is not np.nan])
                wf_sc.write("\t".join(["MEAN", str(mean_mdd), str(mean_ndd)]) + "\n")
                rc.mean_mdd[1] = mean_mdd
                rc.mean_ndd[1] = mean_ndd

                stdev_mdd = stdev([x for x in sc_measurements["second_mdd"] if x is not np.nan])
                stdev_ndd = stdev([x for x in sc_measurements["second_ndd"] if x is not np.nan])
                wf_sc.write("\t".join(["STANDARD DEVIATION", str(stdev_mdd), str(stdev_ndd)]) + "\n")
                rc.stdev_mdd[1] = stdev_mdd
                rc.stdev_ndd[1] = stdev_ndd

        first_df = pd.DataFrame({"MDD": sc_measurements["first_mdd"],
                                 "NDD": sc_measurements["first_ndd"]})
        second_df = pd.DataFrame({"MDD": sc_measurements["second_mdd"],
                                 "NDD": sc_measurements["second_ndd"]})
        # remove any row that contains at least one string value, i.e. a "n/a" value
        first_df.dropna(inplace=True)
        second_df.dropna(inplace=True)

        """# calculate the Wilcoxon signed-rank statistical test and write into the results file
        mdd_stat, mdd_p_value = wilcoxon(list(sc_df["Human_mdd"]), list(sc_df["LLM_mdd"]))
        wf_sc.write("\n===============================================\n")
        wf_sc.write(f"MDD Wilcoxon signed-rank test statistic: {mdd_stat}, p-value: {mdd_p_value}")
        wf_sc.write("\n===============================================\n")

        ndd_stat, ndd_p_value = wilcoxon(list(sc_df["Human_ndd"]), list(sc_df["LLM_ndd"]))
        wf_sc.write("\n===============================================\n")
        wf_sc.write(f"NDD Wilcoxon signed-rank test statistic: {ndd_stat}, p-value: {ndd_p_value}")
        wf_sc.write("\n===============================================\n")"""

    draw_sc_histogram([x for x in sc_measurements["first_ndd"] if not isinstance(x, str)],
                      [x for x in sc_measurements["second_ndd"] if not isinstance(x, str)],
                      rc.mean_ndd[0], rc.mean_ndd[1], mode="NDD")
    draw_sc_histogram([x for x in sc_measurements["first_mdd"] if not isinstance(x, str)],
                      [x for x in sc_measurements["second_mdd"] if not isinstance(x, str)],
                      rc.mean_mdd[0], rc.mean_mdd[1], mode="MDD")