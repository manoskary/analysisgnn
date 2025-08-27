from analysisgnn.utils.graph import *
import partitura as pt
import os
from analysisgnn.data.dataset import BuiltinDataset, StrutturaDataset
from joblib import Parallel, delayed
from analysisgnn.utils import hetero_graph_from_note_array, load_score_hgraph
from tqdm import tqdm
from numpy.lib.recfunctions import structured_to_unstructured
# Removed performance import: from .performance import MatchGraphDataset


class Vienna4x22Dataset(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True):
        url = "https://github.com/CPJKU/vienna4x22"
        super(Vienna4x22Dataset, self).__init__(
            name="Vienna4x22Dataset",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def process(self):
        self.match = list(map(lambda x: os.path.join(self.save_path, "match", x), os.listdir(os.path.join(self.save_path, "match"))))
        self.scores = list(map(lambda x: os.path.join(self.save_path, "musicxml", x), os.listdir(os.path.join(self.save_path, "musicxml"))))
        self.midi = list(map(lambda x: os.path.join(self.save_path, "midi", x), os.listdir(os.path.join(self.save_path, "midi"))))

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True

        return False


class Vienna4x22GraphVoiceSeparationDataset(StrutturaDataset):
    r"""The citation graph dataset, including cora, citeseer and pubmeb.
    Nodes mean authors and edges mean citation relationships.
    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if Vienna4x22 Dataset contining the scores is already available otherwise it will download it.
        Default: ~/.struttura/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """

    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True):
        self.dataset_base = Vienna4x22Dataset(raw_dir=raw_dir)
        if verbose:
            print("Loaded Vienna4x22 Dataset Successfully, now processing...")
        self.graphs = list()
        super(Vienna4x22GraphVoiceSeparationDataset, self).__init__(
            name="Vienna4x22GraphVoiceSeparationDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def process(self):
        pass

    def has_cache(self):

        if os.path.exists(self.save_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        path = os.path.join(self.raw_dir, "Vienna4x22Dataset", "musicxml")
        for fn in os.listdir(path):
            if fn.endswith(".musicxml"):
                g = graph_from_part(os.path.join(path, fn), os.path.splitext(fn)[0])
                g.save(os.path.join(self.save_dir, self.name))
                self.graphs.append(g)

    def load(self):
        self.graphs = list()
        for fn in os.listdir(self.save_path):
            path = os.path.join(self.save_path, fn)
            graph = load_score_graph(path, fn)
            self.graphs.append(graph)

    def __getitem__(self, idx):
        return [[self.graphs[i].x, self.graphs[i].edge_index, self.graphs[i].y] for i in idx]

    def __len__(self):
        return len(self.graphs)

    @property
    def save_name(self):
        return self.name

    @property
    def features(self):
        if self.graphs[0].node_features:
            return self.graphs[0].node_features
        else:
            return list(range(self.graphs[0].x.shape[-1]))


# Commented out due to removed performance dependency
# class Vienna4x22MatchGraphDataset(MatchGraphDataset):
#     """
#     Parameters
#     -----------
#     raw_dir : str
#         Raw file directory to download/contains the input data directory.
#         Dataset will search if Vienna4x22 Dataset contining the scores is already available otherwise it will download it.
#         Default: ~/.struttura/
#     force_reload : bool
#         Whether to reload the dataset. Default: False
#     verbose : bool
#         Whether to print out progress information. Default: True.
#     """
# 
#     def __init__(self, raw_dir=None, force_reload=False,
#                  verbose=True, nprocs=4, **kwargs):
#         dataset_base = Vienna4x22Dataset(raw_dir=raw_dir)
#         super(Vienna4x22MatchGraphDataset, self).__init__(
#             dataset_base=dataset_base,
#             raw_dir=raw_dir,
#             force_reload=force_reload,
#             verbose=verbose,
#             nprocs=nprocs,
#             **kwargs)
