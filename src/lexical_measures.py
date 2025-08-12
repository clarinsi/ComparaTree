from lexical_diversity import lex_div as ld
from statistics import mean, stdev, median

from data_structures import ResultContainer
from general_utils import plot_histogram


"""
    The lexical diversity measures (based on lemmas): 
    Type-token ratio (# of lemma types in corpus / total # of lemmas in corpus)
    Measure of Textual Lexical Diversity, as described in McCarthy and Jarvis (2010)
"""


# define a function that will calculate the lexical diversity measure and plot the histograms
def export_lexical_diversity_measures(first_segments, second_segments, output_dir, rc: ResultContainer):
    print("Exporting lexical diversity measures")

    # open output file
    with open(f"{output_dir}/Lexical_diversity.tsv", "w", encoding="utf-8") as wf_ld:
        print("calculating per-segment LD")

        # prepare list of lemmas for every segment
        first_segms_lemmas = [[tok["lemma"] for sent in s for tok in sent] for s in first_segments]
        second_segms_lemmas = [[tok["lemma"] for sent in s for tok in sent] for s in second_segments]

        # prepare lists to store each ld calculation
        f_ttr_l = list()
        s_ttr_l = list()
        f_mtld_l = list()
        s_mtld_l = list()

        # write ld scores for each dataset
        wf_ld.write(f"========================\n"
                    f"{rc.dataset_names[0]} lexical diversity\n"
                    f"========================\n\n")

        # first write down the header line
        wf_ld.write("\t".join(["SEGM_ID", "TTR", "MTLD"]) + "\n")

        i = 1
        for segm in first_segms_lemmas:
            # calculate ld scores
            f_ttr = ld.ttr(segm)
            f_ttr_l.append(f_ttr)
            f_mtld = ld.mtld(segm)
            f_mtld_l.append(f_mtld)

            wf_ld.write("\t".join([str(i), str(f_ttr), str(f_mtld)]) + "\n")
            i += 1

        wf_ld.write(f"========================\n"
                    f"{rc.dataset_names[1]} lexical diversity\n"
                    f"========================\n\n")
        wf_ld.write("\t".join(["SEGM_ID", "TTR", "MTLD"]) + "\n")

        i = 1
        for segm in second_segms_lemmas:
            s_ttr = ld.ttr(segm)
            s_ttr_l.append(s_ttr)
            s_mtld = ld.mtld(segm)
            s_mtld_l.append(s_mtld)

            wf_ld.write("\t".join([str(i), str(s_ttr), str(s_mtld)]) + "\n")
            i += 1

        wf_ld.write(f"\n\n====================\n"
                    "\t".join(["Average scores", rc.dataset_names[0], rc.dataset_names[1]]) + "\n"
                    f"====================\n")

        # calculate mean ld
        mean_f_ttr = mean(f_ttr_l)
        mean_s_ttr = mean(s_ttr_l)
        wf_ld.write("\t".join(["MEAN TTR", str(mean_f_ttr), str(mean_s_ttr)]) + "\n")
        rc.mean_ttr = [mean_f_ttr, mean_s_ttr]

        # calculate the standard deviation of the ttr score
        stdev_f_ttr = stdev(f_ttr_l) if len(f_ttr_l) > 1 else None
        stdev_s_ttr = stdev(s_ttr_l) if len(s_ttr_l) > 1 else None
        wf_ld.write("\t".join(["STD. DEV. TTR", str(stdev_f_ttr), str(stdev_s_ttr)]) + "\n")
        rc.stdev_ttr = [stdev_f_ttr, stdev_s_ttr]

        """
        # calculate mean mtld
        mean_f_mtld = mean(f_mtld_l)
        mean_s_mtld = mean(s_mtld_l)
        wf_ld.write("\t".join(["MEAN MTLD", str(mean_f_mtld), str(mean_s_mtld)]) + "\n")
        rc.mean_mtld = [mean_f_mtld, mean_s_mtld]

        # calculate the standard deviation of the mtld score
        stdev_f_mtld = stdev(f_mtld_l)
        stdev_s_mtld = stdev(s_mtld_l)
        wf_ld.write("\t".join(["STD. DEV. MTLD", str(stdev_f_mtld), str(stdev_s_mtld)]) + "\n")
        rc.stdev_mtld = [stdev_f_mtld, stdev_s_mtld]

        # calculate the Wilcoxon signed-rank statistical test and write to file
        ttr_stat, ttr_p_value = wilcoxon(f_ttr_l, s_ttr_l)
        wf_ld.write("\n\n===============================================\n")
        wf_ld.write(f"TTR Wilcoxon signed-rank test statistic: {ttr_stat}, p-value: {ttr_p_value}")
        wf_ld.write("\n===============================================")

        mtld_stat, mtld_p_value = wilcoxon(f_mtld_l, s_mtld_l)
        wf_ld.write("\n\n===============================================\n")
        wf_ld.write(f"MTLD Wilcoxon signed-rank test statistic: {mtld_stat}, p-value: {mtld_p_value}")
        wf_ld.write("\n===============================================")"""

        plot_histogram(f_ttr_l, s_ttr_l, "Type-Token Ratio", output_dir, rc, lim_one=True)
        #plot_histogram(f_mtld_l, s_mtld_l, "Measure of Textual Lexical Diversity", output_dir, rc)