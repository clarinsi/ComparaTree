import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from statistics import mean, stdev, median
from math import log, sqrt, ceil

from data_structures import ResultContainer, ComparisonConfig
from general_utils import plot_histogram, draw_violinplot
from stat_utils import return_bootstrapped


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
def export_syntactic_complexity_measure(first_segments: list, second_segments: list, cc: ComparisonConfig, rc: ResultContainer):
    output_dir = cc.output_dir
    segmentation_mode = cc.segmentation_mode

    print(f"Calculating syntactic complexity (MDD and NDD)")

    with open(f"{output_dir}/syntactic_complexity.tsv", "w", encoding="utf-8") as wf_sc:
        sc_measurements = defaultdict(list)
        sc_segment_measurements = defaultdict(list)

        for dataset in [first_segments, second_segments]:
            # introduce dataset and write header
            if dataset == first_segments:
                wf_sc.write(f"=============================================\n"
                            f"{rc.dataset_names[0]} SYNTACTIC COMPLEXITY MEASURES:\n"
                            f"=============================================\n\n")
                wf_sc.write("\t".join(["SENT_ID", "MDD", "NDD"]) + "\n")
            if dataset == second_segments:
                wf_sc.write(f"\n=============================================\n"
                            f"{rc.dataset_names[1]} SYNTACTIC COMPLEXITY MEASURES:\n"
                            f"=============================================\n\n")
                wf_sc.write("\t".join(["SENT_ID", "MDD", "NDD"]) + "\n")

            # calculate sc measures
            i = 0
            for segment in dataset:
                curr_segment_mdd_measurements = list()
                curr_segment_ndd_measurements = list()

                for sent in segment:
                    i += 1
                    sent_mdd = mdd(sent)
                    sent_ndd = ndd(sent, sent_mdd)
                    if dataset == first_segments:
                        sc_measurements["first_mdd"].append(sent_mdd)
                        sc_measurements["first_ndd"].append(sent_ndd)
                    else:
                        sc_measurements["second_mdd"].append(sent_mdd)
                        sc_measurements["second_ndd"].append(sent_ndd)
                    
                    curr_segment_mdd_measurements.append(sent_mdd)
                    curr_segment_ndd_measurements.append(sent_ndd)

                    wf_sc.write("\t".join([str(i), str(sent_mdd), str(sent_ndd)]) + "\n")
                
                if dataset == first_segments:
                    sc_segment_measurements["first_mdd"].append(curr_segment_mdd_measurements) if not all([x is np.nan for x in curr_segment_mdd_measurements]) else sc_segment_measurements["first_mdd"].append(None)
                    sc_segment_measurements["first_ndd"].append(curr_segment_ndd_measurements) if not all([x is np.nan for x in curr_segment_ndd_measurements]) else sc_segment_measurements["first_ndd"].append(None)
                else:
                    sc_segment_measurements["second_mdd"].append(curr_segment_mdd_measurements) if not all([x is np.nan for x in curr_segment_mdd_measurements]) else sc_segment_measurements["second_mdd"].append(None)
                    sc_segment_measurements["second_ndd"].append(curr_segment_ndd_measurements) if not all([x is np.nan for x in curr_segment_ndd_measurements]) else sc_segment_measurements["second_ndd"].append(None)

            # calculate the global mean and stdev values
            if dataset == first_segments:
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

        """
        first_df = pd.DataFrame({"MDD": sc_measurements["first_mdd"],
                                 "NDD": sc_measurements["first_ndd"]})
        second_df = pd.DataFrame({"MDD": sc_measurements["second_mdd"],
                                 "NDD": sc_measurements["second_ndd"]})
        # remove any row that contains at least one string value, i.e. a "n/a" value
        first_df.dropna(inplace=True)
        second_df.dropna(inplace=True)

        # calculate the Wilcoxon signed-rank statistical test and write into the results file
        mdd_stat, mdd_p_value = wilcoxon(list(sc_df["Human_mdd"]), list(sc_df["LLM_mdd"]))
        wf_sc.write("\n===============================================\n")
        wf_sc.write(f"MDD Wilcoxon signed-rank test statistic: {mdd_stat}, p-value: {mdd_p_value}")
        wf_sc.write("\n===============================================\n")

        ndd_stat, ndd_p_value = wilcoxon(list(sc_df["Human_ndd"]), list(sc_df["LLM_ndd"]))
        wf_sc.write("\n===============================================\n")
        wf_sc.write(f"NDD Wilcoxon signed-rank test statistic: {ndd_stat}, p-value: {ndd_p_value}")
        wf_sc.write("\n===============================================\n")"""

    """
    # calculate segment mean values if needed
    sc_segment_measurements["first_mdd_means"] = [mean([x for x in segment if x is not np.nan]) if segment else np.nan for segment in sc_segment_measurements["first_mdd"]]
    sc_segment_measurements["first_ndd_means"] = [mean([x for x in segment if x is not np.nan]) if segment else np.nan for segment in sc_segment_measurements["first_ndd"]]
    sc_segment_measurements["second_mdd_means"] = [mean([x for x in segment if x is not np.nan]) if segment else np.nan for segment in sc_segment_measurements["second_mdd"]]
    sc_segment_measurements["second_ndd_means"] = [mean([x for x in segment if x is not np.nan]) if segment else np.nan for segment in sc_segment_measurements["second_ndd"]]
    """
    
    plot_histogram(sc_measurements["first_mdd"], sc_measurements["second_mdd"], "Mean Dependency Distance", cc, rc)
    
    plot_histogram(sc_measurements["first_ndd"], sc_measurements["second_ndd"], "Normalized Dependency Distance", cc, rc)

    if cc.export_violin_plots:
        draw_violinplot(sc_measurements["first_mdd"], sc_measurements["second_mdd"], "Mean Dependency Distance", cc, rc)
        draw_violinplot(sc_measurements["first_ndd"], sc_measurements["second_ndd"], "Normalized Dependency Distance", cc, rc)

    # bootstrap visualizations
    if cc.export_bootstrapped:
        plot_histogram(return_bootstrapped(sc_measurements["first_ndd"], np.std), return_bootstrapped(sc_measurements["second_ndd"], np.std), "Bootstrapped Normalized Dependency Distance", cc, rc)
    
    #rc.addto_seg_values_df("NDD", sc_segment_measurements["first_ndd_means"], "first")
    #rc.addto_seg_values_df("NDD", sc_segment_measurements["second_ndd_means"], "second")
