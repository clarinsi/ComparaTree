import os
from argparse import ArgumentParser
from statistics import stdev

from data_structures import ResultContainer, ComparisonConfig
from general_utils import split_into_segm, basic_stats_segments, plot_histogram, draw_stripplot, draw_stripplot_single_dataset, write_html_summary, parse_conllu, format_results, get_segm_ids, draw_violinplot
from lexical_measures import export_lexical_diversity_measures
from tag_proportions import export_tag_proportions
from syntactic_complexity import export_syntactic_complexity_measure
from n_grams import export_ngrams
from syntactic_diversity import get_tree_diversity
from stat_utils import return_bootstrapped


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument("--first_file", type=str, required=True,
                           help="Path to the first CoNLL-U file.")
    argparser.add_argument("--second_file", type=str, required=True,
                           help="Path to the second CoNLL-U file.")
    argparser.add_argument("--output_dir", type=str, required=True, 
                           help="Path to the output directory.")
    argparser.add_argument("--analysis_levels", type=str, default="bs,ld,tp,sc",
                           help="List of levels of linguistic analysis that the comparison will be performed on. Separate the values by commas, as such: bs,ld,tp,sc " \
                           "The abbreviations are as follows: bs – basic level, ld – lexical diversity, tp – tag proportions, sc - syntactic complexity, nd – n-gram diversity, sd – syntactic diversity. " \
                           "Pass in 'full' as one of the values for a full analysis.")
    argparser.add_argument("--segmentation_mode", type=str, default="n", choices=["n", "doc_from_newdoc", "doc_from_id"],
                           help="The type of unit to take as a segment. Pass in 'n' to manually set the segment length and specify the length using the " \
                           "'--segment_length' argument. Pass in 'doc_from_newdoc' to split the treebank into documents and infer document boundaries " \
                           "from the 'newdoc id' sentence comment. " \
                           "Pass in 'doc_from_id' to split the treebank into documents and infer document boundaries from sentence ids " \
                           "(assuming that sentence ids are of the format 'doc3022.27.7', where 'doc3022' denotes the document id).")
    argparser.add_argument("--random_segmentation", action="store_true",
                           help="Causes the segmentation process to assign sentences to segments at random. Leave out this argument to assign sentences " \
                           "to segments in sequence without random shuffling.")
    argparser.add_argument("--segment_length", type=int, default=1000,
                           help="Length of segments in number of tokens that the input treebanks will be split into if --segmentation_mode is set to 'n'. Defaults to 1000.")
    argparser.add_argument("--first_treebank_name", type=str, default=None,
                           help="The name to be used for the first treebank. Defaults to the file name of the first dataset.")
    argparser.add_argument("--second_treebank_name", type=str, default=None,
                           help="The name to be used for the second treebank. Defaults to the file name of the second dataset.")
    argparser.add_argument("--stark_config", type=str, default="example_stark_config.ini",
                           help="STARK config file used in the tree extraction process.")
    argparser.add_argument("--n_grams_n_list", type=str, default="3",
                           help="List of values of n for n-gram-based calculations. Separate the values by commas, as such: '3,4,5'.")
    argparser.add_argument("--export_n_grams", action="store_true", 
                           help="Enables exporting n-gram frequencies for every segment. Warning, this can result in very large output files!!")
    
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

    # set dataset names
    if comparison_config.first_dataset_name == comparison_config.second_dataset_name:
        raise Exception("Both datasets currently have the same name, please manually set the dataset names using --first_treebank_name and --second_treebank_name")
    result_container.dataset_names = [comparison_config.first_dataset_name, comparison_config.second_dataset_name]

    # read the annotated files
    first_parsed = parse_conllu(comparison_config.first_file)
    second_parsed = parse_conllu(comparison_config.second_file)

    # split the treebanks into segments
    first_segments = split_into_segm(first_parsed, comparison_config.segmentation_mode, segment_length=comparison_config.segment_length, random_segmentation=comparison_config.random_segmentation)
    second_segments = split_into_segm(second_parsed, comparison_config.segmentation_mode, segment_length=comparison_config.segment_length, random_segmentation=comparison_config.random_segmentation)

    """
    # add the segment ids to the resulting per-segment value dataframe
    result_container.addto_seg_values_df("doc_id", get_segm_ids(first_segments, comparison_config.segmentation_mode), "first")
    result_container.addto_seg_values_df("doc_id", get_segm_ids(second_segments, comparison_config.segmentation_mode), "second")
    """
    
    if any([x in comparison_config.analysis_levels for x in ["bs", "full"]]):
        # first calculate basic statistics for each dataset
        basic_stats_segments(first_segments, second_segments, comparison_config, result_container)

    if any([x in comparison_config.analysis_levels for x in ["ld", "full"]]):
        # calculate lexical diversity measures
        export_lexical_diversity_measures(first_segments, second_segments, comparison_config, result_container)

    if any([x in comparison_config.analysis_levels for x in ["tp", "full"]]):
        # calculate POS tag and dependency relation proportions
        export_tag_proportions(first_parsed, second_parsed, comparison_config, "upos", result_container)
        export_tag_proportions(first_parsed, second_parsed, comparison_config, "deprel", result_container)

    if any([x in comparison_config.analysis_levels for x in ["sc", "full"]]):
        # calculate syntactic complexity measures
        export_syntactic_complexity_measure(first_segments, second_segments, comparison_config, result_container)

    if any([x in comparison_config.analysis_levels for x in ["nd", "full"]]):
        first_ngd_dict = export_ngrams(first_segments, f"first", result_container, comparison_config)
        second_ngd_dict = export_ngrams(second_segments,f"second", result_container, comparison_config)

        # plot the n-gram histograms and store per-segment values
        for n in comparison_config.ngrams_n_list:
            plot_histogram(first_ngd_dict[n], second_ngd_dict[n], f"{n}-Gram Diversity Score", comparison_config, result_container)

            if comparison_config.export_violin_plots:
                draw_violinplot(first_ngd_dict[n], second_ngd_dict[n], f"{n}-Gram Diversity Score", comparison_config, result_container)

            # bootstrap visualizations
            if comparison_config.export_bootstrapped:
                plot_histogram(return_bootstrapped(first_ngd_dict[n], stdev), return_bootstrapped(second_ngd_dict[n], stdev),  f"Bootstrapped {n}-Gram Diversity Score", comparison_config, result_container)

            result_container.addto_seg_values_df(f"{n}GD", first_ngd_dict[n], "first")
            result_container.addto_seg_values_df(f"{n}GD", second_ngd_dict[n], "second")

    if any([x in comparison_config.analysis_levels for x in ["sd", "full"]]):
        # run stark for both datasets and obtain the TDS metric for every segment
        get_tree_diversity(first_segments, second_segments, comparison_config, result_container)

    # format numeric values in result
    format_results(result_container)

    # TODO: only output those html sections that correspond to the user's selected analysis levels
    # summarize the results
    write_html_summary(comparison_config.output_dir, result_container)

    # Export a common dataframe containing per-segment values for various measures
    result_container.export_seg_values_df(os.path.join(comparison_config.output_dir, f"per-segment_df_first.tsv"), "first")
    result_container.export_seg_values_df(os.path.join(comparison_config.output_dir, f"per-segment_df_second.tsv"), "second")

