import os


# A class that stores the most important results of the analysis. The variables are generally in the form of a
# list [first, second], where first refers to the value of a measure in relation to the first corpus
# being compared, while second refers to the value of a measure in relation to the second corpus being compared.
class ResultContainer:
    def __init__(self):
        # dataset names
        self.dataset_names = ["First", "Second"]

        # basic stats
        self.words = [0, 0]
        self.sents = [0, 0]
        self.meanwords = [0, 0]
        self.stdevwords = [0, 0]

        # lexical diversity measures
        self.mean_ttr = [0, 0]
        self.stdev_ttr = [0, 0]
        self.mean_mtld = [0, 0]
        self.stdev_mtld = [0, 0]

        # POS and deprel proportion analysis
        self.pos_top_4 = ["", ""]
        self.deprel_top_4 = ["", ""]

        # syntactic complexity
        self.mean_mdd = [0, 0]
        self.stdev_mdd = [0, 0]
        self.mean_ndd = [0, 0]
        self.stdev_ndd = [0, 0]

        # n-gram analysis
        self.mean_rf_first = dict()
        self.mean_rf_second = dict()
        self.median_rf_first = dict()
        self.median_rf_second = dict()
        self.mean_ngd_first = dict()
        self.stdev_ngd_first = dict()
        self.mean_ngd_second = dict()
        self.stdev_ngd_second = dict()

        # syntactic diversity
        self.tree_diversity_score = [0, 0]
        self.stdev_tree_diversity_score = [0, 0]
        self.unique_trees = [0, 0]
        self.intersection = 0


# A class that stores the configuration for the comparison being made
class ComparisonConfig:
    def __init__(self, args):
        self.first_file = os.path.normpath(args["first_file"])
        self.second_file = os.path.normpath(args["second_file"])
        self.output_dir = os.path.normpath(args["output_dir"])
        self.segment_length = args["segment_length"]
        self.stark_config_file_first = args["first_stark_config"]
        self.stark_config_file_second = args["second_stark_config"]
        self.ngrams_ignore = [",", ".", "?", "!", "*", "=", "#", "∼", "<", ">", "|", "+", "-", "ˆ", "@", "/", "\\", "§"]
        self.ngrams_n_list = [int(x) for x in args["n_grams_n_list"].split(",")]
        self.ngram_threshold = 20
        
        if not args["first_treebank_name"]:
            self.first_dataset_name = self.first_file.split(".")[0]
        else:
            self.first_dataset_name = args["first_treebank_name"]
        
        if not args["second_treebank_name"]:
            self.second_dataset_name = self.second_file.split(".")[0]
        else:
            self.second_dataset_name = args["second_treebank_name"]

        self.analysis_levels = args["analysis_levels"].split(",")
        for lev in self.analysis_levels:
            if lev not in ["bs", "ld", "tp", "sc", "nd", "sd"]:
                raise Exception("Only the following analysis level abbreviations are permitted: bs – basic level, ld – lexical diversity, tp – tag proportions, sc - syntactic complexity, nd – n-gram diversity, sd – syntactic diversity")
