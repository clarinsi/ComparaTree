import os
from argparse import ArgumentParser

from data_structures import ResultContainer, ComparisonConfig
from general_utils import split_into_segm, basic_stats, plot_histogram, write_html_summary, parse_conllu
from lexical_diversity import export_lexical_diversity_measures
from tag_proportions import export_tag_proportions
from syntactic_complexity import export_syntactic_complexity_measure
from n_grams import export_ngrams
from syntactic_diversity import get_tree_diversity


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument("--first_file", type=str, required=True,
                           help="Path to the first CoNLL-U file.")
    argparser.add_argument("--second_file", type=str, required=True,
                           help="Path to the second CoNLL-U file.")
    argparser.add_argument("--output_dir", type=str, required=True, 
                           help="Path to the output directory.")
    argparser.add_argument("--analysis_levels", type=str, default="bs,ld,tp,sc",
                           help="List of levels of linguistic analysis that the comparison will be performed on. Separate the values by commas, as such: 'bs,ld,tp,sc'. " \
                           "The abbreviations are as follows: bs – basic level, ld – lexical diversity, tp – tag proportions, sc - syntactic complexity, nd – n-gram diversity, sd – syntactic diversity")
    argparser.add_argument("--segment_length", type=int, default=1000,
                           help="Length of segments in number of sentences that the input treebanks will be split into.")
    argparser.add_argument("--first_treebank_name", type=str, default=None,
                           help="The name of the first treebank. Defaults to the filename of the first dataset")
    argparser.add_argument("--second_treebank_name", type=str, default=None,
                           help="The name of the second treebank. Defaults to the filename of the second dataset")
    argparser.add_argument("--first_stark_config", type=str,
                           help="Config file for the first treebank.")
    argparser.add_argument("--second_stark_config", type=str,
                           help="Config file for the first treebank.")
    argparser.add_argument("--n_grams_n_list", type=str, default="3",
                           help="List of values of n for n-gram-based calculations. Separate the values by commas, as such: '3,4,5'")
    
    return argparser.parse_args()


if __name__ == "__main__":
    # prepare the configuration structure
    args = parse_args()
    comparison_config = ComparisonConfig(vars(args))

    # check if output dir exists
    if not os.path.isdir(comparison_config.output_dir):
        os.mkdir(comparison_config.output_dir)

    # prepare the ResultContainer
    result_container = ResultContainer()
    result_container.dataset_names = [comparison_config.first_dataset_name, comparison_config.second_dataset_name]

    # read the annotated files
    first_parsed = parse_conllu(comparison_config.first_file)
    second_parsed = parse_conllu(comparison_config.second_file)

    # split the treebanks into segments
    first_segments = split_into_segm(first_parsed, comparison_config.segment_length)
    second_segments = split_into_segm(second_parsed, comparison_config.segment_length)

    if "bs" in comparison_config.analysis_levels:
        # first calculate basic statistics for each dataset
        basic_stats(first_parsed, second_parsed, comparison_config.output_dir, result_container)

    if "ld" in comparison_config.analysis_levels:
        # calculate lexical diversity measures
        export_lexical_diversity_measures(first_segments, second_segments, comparison_config.output_dir, result_container)

    if "tp" in comparison_config.analysis_levels:
        # calculate POS tag and dependency relation proportions
        export_tag_proportions(first_parsed, second_parsed, comparison_config.output_dir, "upos", result_container)
        export_tag_proportions(first_parsed, second_parsed, comparison_config.output_dir, "deprel", result_container)

    if "sc" in comparison_config.analysis_levels:
        # calculate syntactic complexity measures
        export_syntactic_complexity_measure(first_parsed, second_parsed, comparison_config.output_dir, result_container)

    if "nd" in comparison_config.analysis_levels:
        #TODO: add the option to not export all n-grams to the argparser
        first_ngd_dict = export_ngrams(first_segments, comparison_config.output_dir, f"first", comparison_config.ngrams_n_list, result_container,
                                    comparison_config, export_all_ngrams=True)
        second_ngd_dict = export_ngrams(second_segments, comparison_config.output_dir, f"second", comparison_config.ngrams_n_list, result_container,
                                    comparison_config, export_all_ngrams=True)

        # plot the n-gram histograms
        for n in comparison_config.ngrams_n_list:
            plot_histogram(first_ngd_dict[n], second_ngd_dict[n], f"{n}-Gram Diversity Score", comparison_config.output_dir, result_container)

    if "sd" in comparison_config.analysis_levels:
        # run stark for both datasets and obtain the TDS metric for every segment
        print("Starting STARK tree extraction")
        get_tree_diversity(first_segments, second_segments, comparison_config.stark_config_file_first, comparison_config.stark_config_file_second,
                        "Tree Diversity Score", comparison_config.output_dir, result_container)

    #TODO: only output those html fields that pertain to the user's selected analysis levels
    # summarize the results
    write_html_summary(comparison_config.output_dir, result_container)

    """
    # draw the n-gram frequency visualisation stripplots
    draw_stripplot(first_ngram_frequencies, second_ngram_frequencies, output_dir, ngrams_n_list, result_container)

    # draw the rank-frequency distribution plot
    for n in ngrams_n_list:
        draw_rankfreq(first_ngram_frequencies, second_ngram_frequencies, output_dir, n, result_container)

    # run stark for both datasets
    print("Starting STARK tree extraction")
    stark_run(stark_read_settings(stark_config_file_first))
    stark_run(stark_read_settings(stark_config_file_second))
    print("Finished STARK tree extraction")

    # calculate the intersection of both sets of trees and the number of unique trees that appear only in one of the two
    # corpora as well as the syntactic tree diversity scores (STD - # of unique trees / total # of trees)
    # TODO: here you have to include the paths to the files as they appear in the STARK config
    compare_unique_trees(f"{output_dir}/stark_results/{result_container.dataset_names[0]}_output.tsv",
                        f"{output_dir}/stark_results/{result_container.dataset_names[1]}_output.tsv",
                        output_dir, result_container)
    """

