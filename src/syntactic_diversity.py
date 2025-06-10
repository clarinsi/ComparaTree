import os
from statistics import mean, stdev
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from tabulate import tabulate

#TODO: implement STARK as a library when it becomes available via pip
from stark.stark import run as stark_run
from stark.stark import read_settings as stark_read_settings
from stark.stark import parse_args as stark_parse_args

from data_structures import ResultContainer
from general_utils import plot_histogram


"""
    Functions to compare the number of unique trees present in each corpus. This includes the intersection of both sets 
    of trees and the number of unique trees that appear in each corpus as well as the tree diversity scores 
    (TDS - # of unique trees / total # of trees)
"""


def get_tree_diversity(first_segments, second_segments, first_strak_config, second_stark_config, mode, output_dir,
                       rc: ResultContainer):
    print("Making syntactic diversity comparisons")
    first_tds_list = list()
    second_tds_list = list()
    num_segments_first = len(first_segments)
    num_segments_second = len(second_segments)

    # first write out every segment into a conllu file
    if not os.path.isdir(f"{output_dir}/syntactic_diversity"):
        os.mkdir(f"{output_dir}/syntactic_diversity")
    if not os.path.isdir(f"{output_dir}/syntactic_diversity/conllu"):
        os.mkdir(f"{output_dir}/syntactic_diversity/conllu")
    if not os.path.isdir(f"{output_dir}/syntactic_diversity/conllu/first"):
        os.mkdir(f"{output_dir}/syntactic_diversity/conllu/first")
    if not os.path.isdir(f"{output_dir}/syntactic_diversity/conllu/second"):
        os.mkdir(f"{output_dir}/syntactic_diversity/conllu/second")

    seg_id = 1
    for first_seg in first_segments:
        with open(f"{output_dir}/syntactic_diversity/conllu/first/segment_{seg_id}.conllu", "w", encoding="utf-8") \
                as rf_first:

            rf_first.write("".join([sent.serialize() for sent in first_seg]))
            seg_id += 1

    seg_id = 1
    for second_seg in second_segments:
        with open(f"{output_dir}/syntactic_diversity/conllu/second/segment_{seg_id}.conllu", "w", encoding="utf-8") \
                as rf_second:
            rf_second.write("".join([sent.serialize() for sent in second_seg]))
            seg_id += 1

    # generate stark configs
    if not os.path.isdir(f"{output_dir}/syntactic_diversity/stark_configs_first"):
        os.mkdir(f"{output_dir}/syntactic_diversity/stark_configs_first")
    if not os.path.isdir(f"{output_dir}/syntactic_diversity/stark_configs_second"):
        os.mkdir(f"{output_dir}/syntactic_diversity/stark_configs_second")
    if not os.path.isdir(f"{output_dir}/syntactic_diversity/stark_trees_first"):
        os.mkdir(f"{output_dir}/syntactic_diversity/stark_trees_first")
    if not os.path.isdir(f"{output_dir}/syntactic_diversity/stark_trees_second"):
        os.mkdir(f"{output_dir}/syntactic_diversity/stark_trees_second")

    with open(first_strak_config, "r", encoding="utf-8") as rf_config:
        base_config = rf_config.read()

    # write the configs
    for seg_id in range(num_segments_first):
        with open(f"{output_dir}/syntactic_diversity/stark_configs_first/config_{str(seg_id + 1)}.ini", "w",
                  encoding="utf-8") as wf_config:
            wf_config.write(base_config.replace("input = ../Datasets/SSJ-UD_v2.15/ssj_merged.conllu",
                                                f"input = {output_dir}/syntactic_diversity/conllu/first/segment_{str(seg_id + 1)}.conllu").replace(
                                                "output = Analysis_2_SSJ_vs_SST/stark_results/SSJ_output.tsv",
                                                f"output = {output_dir}/syntactic_diversity/stark_trees_first/segment_{str(seg_id + 1)}.txt"))

    for seg_id in range(num_segments_second):
        with open(f"{output_dir}/syntactic_diversity/stark_configs_second/config_{str(seg_id + 1)}.ini", "w",
                  encoding="utf-8") as wf_config:
            wf_config.write(base_config.replace("input = ../Datasets/SSJ-UD_v2.15/ssj_merged.conllu",
                                                f"input = {output_dir}/syntactic_diversity/conllu/second/segment_{str(seg_id + 1)}.conllu").replace(
                                                "output = Analysis_2_SSJ_vs_SST/stark_results/SSJ_output.tsv",
                                                f"output = {output_dir}/syntactic_diversity/stark_trees_second/segment_{str(seg_id + 1)}.txt"))

    # run stark
    for seg_id in range(num_segments_first):
        stark_run(stark_read_settings(f"{output_dir}/syntactic_diversity/stark_configs_first/config_{str(seg_id + 1)}.ini"))

    for seg_id in range(num_segments_second):
        stark_run(stark_read_settings(f"{output_dir}/syntactic_diversity/stark_configs_second/config_{str(seg_id + 1)}.ini"))

    # calculate the tds for each treebank
    for seg_id in range(num_segments_first):
        with open(f"{output_dir}/syntactic_diversity/stark_trees_first/segment_{str(seg_id + 1)}.txt", "r",
                  encoding="utf-8") as rf_first_trees:
            trees = list()
            freqs = list()
            for line in rf_first_trees:
                if not line.startswith("Tree\tAbsolute"):
                    tree, freq = line.split("\t")[:2]
                    trees.append(tree)
                    freqs.append(int(freq))

            # calculate the tree diversity score for this segment
            segment_tds = len(trees) / sum(freqs)
            first_tds_list.append(segment_tds)

    for seg_id in range(num_segments_second):
        with open(f"{output_dir}/syntactic_diversity/stark_trees_second/segment_{str(seg_id + 1)}.txt", "r",
                  encoding="utf-8") as rf_second_trees:
            trees = list()
            freqs = list()
            for line in rf_second_trees:
                if not line.startswith("Tree\tAbsolute"):
                    tree, freq = line.split("\t")[:2]
                    trees.append(tree)
                    freqs.append(int(freq))

            # calculate the tree diversity score for this segment
            segment_tds = len(trees) / sum(freqs)
            second_tds_list.append(segment_tds)

    # store the mean and standard deviation of the TDS for each treebank and draw a histogram
    rc.tree_diversity_score = [mean(first_tds_list), mean(second_tds_list)]
    rc.stdev_tree_diversity_score = [stdev(first_tds_list), stdev(second_tds_list)]
    plot_histogram(first_tds_list, second_tds_list, "Tree Diversity Score", output_dir, rc)


def compare_unique_trees(first_trees_file, second_trees_file, output_dir, rc: ResultContainer):
    def draw_venn(Ab, aB, AB):
        fig = plt.figure(figsize=(12, 6))
        venn2(subsets=(Ab, aB, AB), set_labels=(rc.dataset_names[0], rc.dataset_names[1]), set_colors=("green", "pink"))
        plt.title(f"Syntactic Tree Venn Diagram")
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.25)

        # Caption and display the plot
        caption = (r"$\bf{Figure\ 6:}$" + f"Venn diagram comparison of the number of unique syntactic trees present in "
                   f"each treebank.")
        fig.text(0, 0.01, caption, wrap=True, fontsize=10)

        plt.savefig(f"{output_dir}/tree_venn_diagram.png")
    print("Making syntactic tree list comparisons")

    first_trees = list()
    first_freqs = list()
    with open(first_trees_file, "r", encoding="utf-8") as rf_first:
        for line in rf_first.readlines()[1:]:
            tree, freq = line.split("\t")[:2]
            first_trees.append(tree)
            first_freqs.append(int(freq))

    second_trees = list()
    second_freqs = list()
    with open(second_trees_file, "r", encoding="utf-8") as rf_second:
        for line in rf_second.readlines()[1:]:
            tree, freq = line.split("\t")[:2]
            second_trees.append(tree)
            second_freqs.append(int(freq))

    # draw venn diagram
    first_min_second = len(set(first_trees) - set(second_trees))
    second_min_first = len(set(second_trees) - set(first_trees))
    intersection = len(set(first_trees) & set(second_trees))
    draw_venn(first_min_second, second_min_first, intersection)
    rc.unique_trees = [len(set(first_trees)), len(set(second_trees))]
    rc.intersection = intersection

    # calculate the tree diversity score
    first_tds = len(first_trees) / sum(first_freqs)
    second_tds = len(second_trees) / sum(second_freqs)
    rc.tree_diversity_score = [first_tds, second_tds]

    # output summary
    with open(f"{output_dir}/stark_results/tree_comparison.txt", "w", encoding="utf-8") as results_wf:
        results_wf.write(tabulate([["total no. of trees first", sum(first_freqs)],
                                   ["total no. of trees second", sum(second_freqs)],
                                   ["no. of unique trees first", len(first_trees)],
                                   ["no. of unique trees second", len(second_trees)],
                                   ["intersection", intersection],
                                   ["first - second", first_min_second],
                                   ["second - first", second_min_first],
                                   ["first tree diversity", first_tds],
                                   ["second tree diversity", second_tds]],
                                  headers=["", "value"], intfmt=",", tablefmt="rounded_outline"))