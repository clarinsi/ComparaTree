from statistics import mean, stdev
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib_venn import venn2
from tabulate import tabulate
import conllu
import os
import logging

from data_structures import ResultContainer
from stat_utils import check_for_normality


# parse conllu files
def parse_conllu(filepath):
    print(f"Reading treebank from {os.path.split(filepath)[1]}")
    with open(filepath, "r", encoding="utf-8") as rf:
        parsed_sents = conllu.parse(rf.read())
    
    return parsed_sents


# basic segmentation of the treebanks. There are three segmentation modes:
# 1. the "n" mode is the default mode. It splits the treebank into approximately equally sized segments of n tokens.
# 2. the "doc_from_newdoc" mode splits the treebank into documents and tries to infer document boundaries from the "newdoc id" comment 
# 3. the "doc_from_id" mode splits the treebank into documents and tries to infer document boundaries from the sentence id (the assumption is that sentence ids are of the format "doc3022.27.7", where "doc3022" denotes the document id).
def split_into_segm(dataset, segmentation_mode, segment_length=1000):
    assert segmentation_mode in ["n", "doc_from_newdoc", "doc_from_id"]
    segments = list()

    if segmentation_mode == "n":
        curr_segment = list()
        i = 0
        for sent in dataset:
            i += len(sent)
            curr_segment.append(sent)
            if i > segment_length:
                segments.append(curr_segment)
                curr_segment = list()
                i = 0
        
        if len(curr_segment) > 0:
            segments.append(curr_segment)

    elif segmentation_mode == "doc_from_newdoc":
        curr_segment = list()
        for sent in dataset:
            if "newdoc_id" in sent.metadata.keys() and len(segments) > 0:
                segments.append(curr_segment)
                curr_segment = list()
            curr_segment.append(sent)
        
        if len(curr_segment) > 0:
            segments.append(curr_segment)
    
    elif segmentation_mode == "doc_from_id":
        curr_segment = list()
        old_doc = None
        curr_doc = None
        for sent in dataset:
            assert "." in sent.metadata["sent_id"]

            curr_doc = sent.metadata["sent_id"].split(".")[0]
            if not old_doc:
                old_doc = curr_doc
            if curr_doc != old_doc:
                segments.append(curr_segment)
                curr_segment = list()
                old_doc = curr_doc
            curr_segment.append(sent)
        
        if len(curr_segment) > 0:
            segments.append(curr_segment)

    return segments


def get_segm_ids(segments, segmentation_mode):
    ids_list = list()
    if segmentation_mode in ["doc_from_newdoc", "doc_from_id"]:
        for segment in segments:
            ids_list.append(segment[0].metadata["sent_id"].split(".")[0])
    else:
        for i in range(len(segments)):
            ids_list.append(i)
    
    return ids_list


"""
    Visualization functions
"""


# function for drawing histograms. first_data and second_data should be lists of calculated values for the measure for
# every segment
def plot_histogram(first_data, second_data, measure, output_dir, rc: ResultContainer, lim_one=False, segmentation_mode="n"):

    # manually set matplotlib's logging level due to some strange debug messages
    plt.set_loglevel("warning")
    logging.getLogger('PIL').setLevel(logging.WARNING)

    print(f"Plotting {measure} histogram")
    first_mean = mean([x for x in first_data if x is not np.nan])
    second_mean = mean([x for x in second_data if x is not np.nan])
    first_stdev = stdev([x for x in first_data if x is not np.nan]) if len(first_data) > 1 else None
    second_stdev = stdev([x for x in second_data if x is not np.nan]) if len(second_data) > 1 else None

    # make histograms
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot the first histogram
    sns.histplot(first_data, kde=False, ax=axes[0])
    axes[0].set_title(f"Treebank={rc.dataset_names[0]}")
    axes[0].set(xlabel=f"Range of values for {measure}")
    if lim_one:
        axes[0].set_xlim(0, 1)
    axes[0].axvline(x=first_mean, color='red', linestyle='-', linewidth=2)
    axes[0].text(0.95, 0.95,  f"Mean = {first_mean:.3f}", color='red', fontsize=14, ha='right', va='top',
                 transform=axes[0].transAxes)
    if first_stdev:
        axes[0].text(0.95, 0.80, f"Standard deviation = {first_stdev:.3f}", color='orange', fontsize=14, ha='right',
                     va='top', transform=axes[0].transAxes)
    else:
        axes[0].text(0.95, 0.80, f"Standard deviation = N/A", color='orange', fontsize=14, ha='right',
                     va='top', transform=axes[0].transAxes)

    # Plot the second histogram
    sns.histplot(second_data, kde=False, ax=axes[1])
    axes[1].set_title(f"Treebank={rc.dataset_names[1]}")
    axes[1].set(xlabel=f"Range of values for {measure}")
    if lim_one:
        axes[1].set_xlim(0, 1)
    axes[1].axvline(x=second_mean, color='red', linestyle='-', linewidth=2)
    axes[1].text(0.95, 0.95, f"Mean = {second_mean:.3f}", color='red', fontsize=14, ha='right', va='top',
                 transform=axes[1].transAxes)
    if second_stdev:
        axes[1].text(0.95, 0.80, f"Standard deviation = {second_stdev:.3f}", color='orange', fontsize=14, ha='right',
                     va='top', transform=axes[1].transAxes)
    else:
        axes[1].text(0.95, 0.80, f"Standard deviation = N/A", color='orange', fontsize=14, ha='right',
                     va='top', transform=axes[1].transAxes)

    # handle segmentation mode wording
    if segmentation_mode == "n":
        segmentation_mode_wording = "Per-Segment"
    else:
        segmentation_mode_wording = "Per-Document"

    if measure == "Segment Length":
        segmentation_mode_wording = "" # Special case

    # set the title and add a caption
    fig.suptitle(f"{segmentation_mode_wording} {measure} Histogram")

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    caption = (r"$\bf{Figure:}$" + f"Histogram showing the frequency distribution for the {segmentation_mode_wording.lower()} "
                f"{measure} in both treebanks. The x axis represents the range of values for the {measure}. "
                f"The blue bars represent the number of observations for each value "
                f"of the measure. The red vertical line represents the mean of the {measure}.")
    fig.text(0, 0.01, caption, wrap=True, fontsize=10)

    plt.savefig(f"{output_dir}/{measure}_histogram.png")


# general function for drawing stripplots
def draw_stripplot(first_data, second_data, measure, output_dir, names):

    # manually set matplotlib's logging level due to some strange debug messages
    plt.set_loglevel("warning")
    logging.getLogger('PIL').setLevel(logging.WARNING)

    print(f"Plotting {measure} stripplot")
    stripplot_df = pd.DataFrame({"value": list(first_data) + list(second_data),
                                 "dataset": [names[0]] * len(first_data) + [names[1]] * len(second_data)})

    plt.figure(figsize=(10, 8))
    sns.stripplot(data=stripplot_df, x="dataset", y="value", jitter=True, alpha=0.6, palette="flare")
    plt.title(f"{measure} Stripplot")
    plt.savefig(os.path.join(output_dir, f"{measure}_stripplot.png"))
    plt.close()


# function for drawing a stripplot for a single dataset for all values
def draw_stripplot_single_dataset(dataset, measures, output_dir, dataset_name):
    n = len(measures)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 6), sharey=False)

    # If there's only one measure, axes won't be a list
    if n == 1:
        axes = [axes]

    for ax, measure in zip(axes, measures):
        sns.stripplot(y=dataset[measure], ax=ax, jitter=True, alpha=0.6)
        ax.set_title(measure)
        ax.set_xlabel("")
        ax.text(0.5, -0.15, f"Shapiro-Wilk p value: {check_for_normality(dataset[measure])[1]}", 
                ha="center", va="top", transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_combined_stripplots.png"))
    plt.close()


# output the basic statistics for a dataset composed of sentences in the conllu format
def basic_stats(first_dataset, second_dataset, output_dir, rc: ResultContainer):
    print("Calculating basic statistics")
    first_total_words = len([word for sent in first_dataset for word in sent])
    first_total_sents = len(first_dataset)

    second_total_words = len([word for sent in second_dataset for word in sent])
    second_total_sents = len(second_dataset)

    # mean number of words per sentence
    first_wps = [len(sent) for sent in first_dataset]
    second_wps = [len(sent) for sent in second_dataset]
    first_mean_words = mean(first_wps)
    second_mean_words = mean(second_wps)
    # standard deviation of words per sentence
    first_stdev_words = stdev([len(sent) for sent in first_dataset])
    second_stdev_words = stdev([len(sent) for sent in second_dataset])

    basic_stats_table = "---------------\nBasic statistics for the two datasets:\n---------------\n\n" + tabulate(
        [["First", first_total_words, first_total_sents, first_mean_words, first_stdev_words],
         ["Second", second_total_words, second_total_sents, second_mean_words, second_stdev_words]],
        headers=["Dataset", "Words total", "Sentences total", "Mean words per sentence",
                 "Stdev words per sentence"], intfmt=",", tablefmt="rounded_outline") + "\n---------------\n\n"
    with open(f"{output_dir}/basic_stats.txt", "w", encoding="utf-8") as wf_basic:
        wf_basic.write(basic_stats_table)

    rc.words = [first_total_words, second_total_words]
    rc.sents = [first_total_sents, second_total_sents]
    rc.meanwords = [first_mean_words, second_mean_words]
    rc.stdevwords = [first_stdev_words, second_stdev_words]

    plot_histogram(first_wps, second_wps, "Average Sentence Length", output_dir, rc)


# tweaked version of the basic statistics calculation function which works with segments
def basic_stats_segments(first_segments, second_segments, output_dir, rc: ResultContainer, segmentation_mode="n"):
    print("Calculating basic statistics")
    first_total_words = len([word for first_seg in first_segments for sent in first_seg for word in sent])
    first_total_sents = len([sent for first_seg in first_segments for sent in first_seg])
    first_total_segs = len(first_segments)

    second_total_words = len([word for second_seg in second_segments for sent in second_seg for word in sent])
    second_total_sents = len([sent for second_seg in second_segments for sent in second_seg])
    second_total_segs = len(second_segments)

    # mean number of words per sentence averaged over segments
    first_seg_lengths = [[len(sent) for sent in first_seg] for first_seg in first_segments]
    second_seg_lengths = [[len(sent) for sent in second_seg] for second_seg in second_segments]
    first_seg_mean_words = [mean(seg_lengths) for seg_lengths in first_seg_lengths]
    second_seg_mean_words = [mean(seg_lengths) for seg_lengths in second_seg_lengths]

    # overall segment lengths
    first_overall_seg_lengths = [sum(seg_lengths) for seg_lengths in first_seg_lengths]
    second_overall_seg_lengths = [sum(seg_lengths) for seg_lengths in second_seg_lengths]

    # mean number of words per sentence for each treebank in full
    first_wps = [length for seg_lengths in first_seg_lengths for length in seg_lengths]
    second_wps = [length for seg_lengths in second_seg_lengths for length in seg_lengths]

    first_mean_words = mean(first_wps)
    second_mean_words = mean(second_wps)

    first_stdev_words = stdev(first_wps)
    second_stdev_words = stdev(second_wps)

    basic_stats_table = "---------------\nBasic statistics for the two datasets:\n---------------\n\n" + tabulate(
        [["First", first_total_words, first_total_sents, first_total_segs, first_mean_words, first_stdev_words],
         ["Second", second_total_words, second_total_sents, second_total_segs, second_mean_words, second_stdev_words]],
        headers=["Dataset", "Words total", "Sentences total", "Segments total", "Mean words per sentence", "Stdev words per sentence"], intfmt=",", tablefmt="rounded_outline") + "\n---------------\n\n"
    with open(f"{output_dir}/basic_stats.txt", "w", encoding="utf-8") as wf_basic:
        wf_basic.write(basic_stats_table)

    rc.words = [first_total_words, second_total_words]
    rc.sents = [first_total_sents, second_total_sents]
    rc.meanwords = [first_mean_words, second_mean_words]
    rc.stdevwords = [first_stdev_words, second_stdev_words]

    # visualization of overall segment length. This excludes cases when the segmentation mode is set to n, 
    # since segment length is obvious in that case
    if segmentation_mode != "n":
        plot_histogram(first_overall_seg_lengths, second_overall_seg_lengths, "Segment Length", output_dir, rc, segmentation_mode=segmentation_mode)

    # segment level visualization of mean sentence length
    #plot_histogram(first_seg_mean_words, second_seg_mean_words, "Average Sentence Length", output_dir, rc, segmentation_mode=segmentation_mode)

    # sentence level visualization of mean sentence length
    plot_histogram(first_wps, second_wps, "Average Sentence Length", output_dir, rc)

    rc.addto_seg_values_df("seg_length", first_overall_seg_lengths, "first")
    rc.addto_seg_values_df("seg_length", second_overall_seg_lengths, "second")


"""
    Functions to summarize all the results
"""


def format_results(rc:ResultContainer):
    for x in [rc.meanwords[0], rc.meanwords[1], rc.stdevwords[0], rc.stdevwords[1], rc.mean_ttr[0], rc.mean_ttr[1],
              rc.stdev_ttr[0], rc.stdev_ttr[1], rc.mean_mdd[0], rc.mean_mdd[1], rc.stdev_mdd[0], rc.stdev_mdd[1],
              rc.mean_ndd[0], rc.mean_ndd[1], rc.stdev_ndd[0], rc.stdev_ndd[1], rc.tree_diversity_score[0], 
              rc.tree_diversity_score[1], rc.stdev_tree_diversity_score[0], rc.stdev_tree_diversity_score[1]]:
        x = f"{x:.3f}" if x else "N/A"


    for n in rc.mean_ngd_first.keys():
        for x in [rc.mean_ngd_first[n], rc.mean_ngd_second[n], rc.stdev_ngd_first[n], rc.stdev_ngd_second[n]]:
            x = f"{x:.3f}" if x else "N/A"


def summarize(output_dir, rc: ResultContainer):
    print("Generating summary")
    data_to_tabulate = [["Total # of words", rc.words[0], rc.words[1]],
                        ["Total # of sentences", rc.sents[0], rc.sents[1]],
                        ["Average # of tokens per sentence", rc.meanwords[0], rc.meanwords[1]],
                        ["---------------------------------------------", "---------------------------------------------", "---------------------------------------------"],
                        ["Mean Type-Token Ratio", rc.mean_ttr[0], rc.mean_ttr[1]],
                        ["Mean Textual Lexical Diversity score", rc.mean_mtld[0], rc.mean_mtld[1]],
                        ["---------------------------------------------", "---------------------------------------------", "---------------------------------------------"],
                        ["4 most typical UPOS tag proportions", rc.pos_top_4[0], rc.pos_top_4[1]],
                        ["4 most typical dependency relation proportions", rc.deprel_top_4[0], rc.deprel_top_4[1]],
                        ["---------------------------------------------", "---------------------------------------------", "---------------------------------------------"],
                        ["Average Mean Dependency Distance", rc.mean_mdd[0], rc.mean_mdd[1]],
                        ["Average Normalized Dependency Distance", rc.mean_ndd[0], rc.mean_ndd[1]],
                        ["---------------------------------------------", "---------------------------------------------", "---------------------------------------------"],
                        ["Syntactic Tree Diversity Score", rc.tree_diversity_score[0], rc.tree_diversity_score[1]],
                        ["# of unique syntactic trees", rc.unique_trees[0], rc.unique_trees[1]],
                        [f"Intersection of unique syntactic trees: {rc.intersection}", "---------------------------------------------", "---------------------------------------------"],
                        ["---------------------------------------------", "---------------------------------------------", "---------------------------------------------"],]

    for n in rc.mean_rf_first.keys():
        data_to_tabulate.append([f"{n}-gram mean frequency per million", rc.mean_rf_first[n], rc.mean_rf_second[n]])

    for n in rc.median_rf_first.keys():
        data_to_tabulate.append([f"{n}-gram median frequency per million", rc.median_rf_first[n], rc.median_rf_second[n]])

    for n in rc.ngd_first.keys():
        data_to_tabulate.append([f"{n}-gram diversity score", rc.ngd_first[n], rc.ngd_second[n]])

    with open(f"{output_dir}/Results_summary.txt", "w", encoding="utf-8") as wf:
        wf.write(tabulate(data_to_tabulate, headers=["Metric", rc.dataset_names[0], rc.dataset_names[1]], intfmt=",",
                          tablefmt="rounded_outline"))


def write_html_summary(output_dir, rc: ResultContainer):
    print("Generating HTML summary")
    ngram_cells = """
        <tr>
            <td colspan="3" style="text-align: center; background-color: #ddfff1"><strong>N-Gram Diversity</strong></td>
        </tr>"""
    for n in rc.mean_ngd_first.keys():
        ngram_cells += f"""
        <tr>
            <td>Average Segmental {n}-gram Diversity Score</td>
            <td>{rc.mean_ngd_first[n]}</td>
            <td>{rc.mean_ngd_second[n]}</td>
        </tr>
        <tr>
            <td>Segmental {n}-gram Diversity Score standard deviation</td>
            <td>{rc.stdev_ngd_first[n]}</td>
            <td>{rc.stdev_ngd_second[n]}</td>
        </tr>"""

    table_html = f"""
    <table>
        <tr>
            <th style="background-color: #e7e7ff">Metric</th>
            <th style="background-color: #e7e7ff">{rc.dataset_names[0]}</th>
            <th style="background-color: #e7e7ff">{rc.dataset_names[1]}</th>
        </tr>
        <tr>
            <td colspan="3" style="text-align: center; background-color: #ddfff1"><strong>Basic</strong></td>
        </tr>
        <tr>
            <td>Total # of tokens</td>
            <td>{rc.words[0]:,}</td>
            <td>{rc.words[1]:,}</td>
        </tr>
        <tr>
            <td>Total # of sentences</td>
            <td>{rc.sents[0]:,}</td>
            <td>{rc.sents[1]:,}</td>
        </tr>
        <tr>
            <td>Average tokens per sentence</td>
            <td>{rc.meanwords[0]}</td>
            <td>{rc.meanwords[1]}</td>
        </tr>
        <tr>
            <td>Standard deviation of tokens per sentence</td>
            <td>{rc.stdevwords[0]}</td>
            <td>{rc.stdevwords[1]}</td>
        </tr>
        <tr>
            <td colspan="3" style="text-align: center; background-color: #ddfff1"><strong>Lexical Diversity</strong></td>
        </tr>
        <tr>
            <td>Average Segmental Type-Token Ratio</td>
            <td>{rc.mean_ttr[0]}</td>
            <td>{rc.mean_ttr[1]}</td>
        </tr>
        <tr>
            <td>Segmental Type-Token Ratio standard deviation</td>
            <td>{rc.stdev_ttr[0]}</td>
            <td>{rc.stdev_ttr[1]}</td>
        </tr>
        {ngram_cells}
        <tr>
            <td colspan="3" style="text-align: center; background-color: #ddfff1"><strong>UD Label Proportions</strong></td>
        </tr>
        <tr>
            <td>Largest part-of-speech tag proportion differences</td>
            <td>{rc.pos_top_4[0]}</td>
            <td>{rc.pos_top_4[1]}</td>
        </tr>
        <tr>
            <td>Largest dependency relation proportion differences</td>
            <td>{rc.deprel_top_4[0]}</td>
            <td>{rc.deprel_top_4[1]}</td>
        </tr>
        <tr>
            <td colspan="3" style="text-align: center; background-color: #ddfff1"><strong>Syntactic Complexity</strong></td>
        </tr>
        <tr>
            <td>Average Mean Dependency Distance</td>
            <td>{rc.mean_mdd[0]}</td>
            <td>{rc.mean_mdd[1]}</td>
        </tr>
        <tr>
            <td>Mean Dependency Distance standard deviation</td>
            <td>{rc.stdev_mdd[0]}</td>
            <td>{rc.stdev_mdd[1]}</td>
        </tr>
        <tr>
            <td>Average Normalized Dependency Distance</td>
            <td>{rc.mean_ndd[0]}</td>
            <td>{rc.mean_ndd[1]}</td>
        </tr>
        <tr>
            <td>Normalized Dependency Distance standard deviation</td>
            <td>{rc.stdev_ndd[0]}</td>
            <td>{rc.stdev_ndd[1]}</td>
        </tr>
        <tr>
            <td colspan="3" style="text-align: center; background-color: #ddfff1"><strong>Syntactic Diversity</strong></td>
        </tr>
        <tr>
            <td>Average Segmental Tree Diversity Score</td>
            <td>{rc.tree_diversity_score[0]}</td>
            <td>{rc.tree_diversity_score[1]}</td>
        </tr>
        <tr>
            <td>Segmental Tree Diversity Score standard deviation</td>
            <td>{rc.stdev_tree_diversity_score[0]}</td>
            <td>{rc.stdev_tree_diversity_score[1]}</td>
        </tr>"""

    table_html += """
    </table>"""

    ngram_plots = ""
    for n in rc.mean_ngd_first.keys():
        ngram_plots += f'<img style="margin-top: 40pt" src="{n}-Gram Diversity Score_histogram.png"><br>\n'

    html_content = """
    <html>
    <head>
        <title>Comparison Summary</title>
        <style>
        table, th, td {
          border: 1px solid black;
        }
        </style>
    </head>""" + f"""
    <body>
        <h1>{rc.dataset_names[0]} vs {rc.dataset_names[1]} Comparison Summary</h1>
        <h2>Results Table</h2>
            {table_html}
        <h2>Visualizations</h2>
        <div>
            <h3>Basic</h3>
                <img src="Average Sentence Length_histogram.png"><br>
        </div>
        <div>
            <h3>Lexical Diversity</h3>
                <img src="Type-Token Ratio_histogram.png"><br>
        </div>
        <div>
            <h3 style="margin-top: 40pt">N-Gram Diversity</h3>
                {ngram_plots}
        </div>
        <div>
            <h3 style="margin-top: 40pt">UD Label Proportions</h3>
                <img src="upos_barcharts.png"><br>
                <img style="margin-top: 40pt" src="deprel_barcharts.png"><br>
        </div>
        <div>
            <h3 style="margin-top: 40pt">Syntactic Complexity</h3>
                <img src="Normalized Dependency Distance_histogram.png"><br>
        </div>
        <div>
            <h3 style="margin-top: 40pt">Syntactic Diversity</h3>
                <img src="Tree Diversity Score_histogram.png"><br>
        </div>
    </body>
    </html>
    """

    with open(f"{output_dir}/result_summary.html", "w", encoding="utf-8") as wf:
        wf.write(html_content)
