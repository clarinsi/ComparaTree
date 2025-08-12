# ComparaTree

ComparaTree is a tool for generating automatic treebank comparisons of treebanks in the CoNLL-U format. The tool calculates various metrics and generates visualizations that illustrate the similarities and differences between two treebanks. The tool can generate comparisons along several dimensions of linguistic analysis: lexical diversity, n-gram diversity, part-of-speech and dependency relation proportions, syntactic complexity, and syntactic diversity.

It expects two treebank files in the CoNLL-U format on input and outputs an HTML summary of the comparison. ComparaTree is primarily intended to be used with treebanks conforming to the Universal Dependencies formalism, however in principle any dependency grammar formalism that uses the CoNLL-U file encoding should be supported.

## Installation

To install the necessary requirements for ComparaTree, navigate to the ComparaTree folder and create a new conda environment using the environment.yml file:
```
conda env create -f environment.yml
```
This will create a conda environment named `ComparaTree_env` with everything that is needed to run comparisons.

## Basic usage

To run a comparison using two treebanks in the CoNLL-U format, run the following command:

```
python src/Run_analysis.py --first_file path/to/first/file.conllu --second_file path/to/second/file.conllu --output_dir path/to/output/directory
```

This will generate the comparison results for lexical diversity, tag proportions, and syntactic complexity and store them in the output directory (including calculations for measures, visualizations, HTML summary). See the `Settings` section below for more details on how to run the analysis for other linguistic analysis levels and adjust various other configuration settings.

## Settings

Other configuration options that can be used when running the command described above:

* `--analysis_levels`: List of levels of linguistic analysis that the comparison will be performed on. Separate the values by commas, as such: 'bs,ld,tp,sc'. The abbreviations are as follows: bs – basic level, ld – lexical diversity, tp – tag proportions, sc - syntactic complexity, nd – n-gram diversity, sd – syntactic diversity. Pass in 'full' as one of the values for a full analysis. Defaults to 'bs,ld,tp,sc'.

NOTE: The syntactic diversity measure calculations currently incorporate the external [STARK](https://github.com/clarinsi/STARK) tool for dependency tree extraction. The process can be resource intensive and time consuming when generating comparisons for large treebanks. It is recommended that you set the segment length (see below) to at least 1000 (or more) when incorporating syntactic diversity comparisons for large treebanks.

* `--segment_length`: Length of segments (in number of tokens) that the input treebanks will be split into. The segments are used in calculating lexical diversity, n-gram diversity, and syntactic diversity measures. Defaults to 1000.

* `--first_treebank_name`: The name to be used for the first treebank. Defaults to the file name of the first dataset.

* `--second_treebank_name`: The name to be used for the second treebank. Defaults to the file name of the second dataset.

* `--stark_config`: STARK config file used in the tree extraction process.

* `--n_grams_n_list`: List of values of n for n-gram-based calculations. Separate the values by commas, as such: '3,4,5'. Defaults to '3'.

* `--export_n_grams`: Enables exporting n-gram frequencies for every segment. Warning, this can result in very large output files!!
