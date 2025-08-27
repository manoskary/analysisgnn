from graphmuse import create_score_graph
import numpy as np
from analysisgnn.utils import load_score_hgraph, hetero_graph_from_note_array, select_features, HeteroScoreGraph
from analysisgnn.utils import time_divided_tsv_to_part
import torch
import os
from analysisgnn.data.dataset import BuiltinDataset, StrutturaDataset, get_download_dir
from joblib import Parallel, delayed
from tqdm import tqdm
import random
from analysisgnn.utils.music import PitchEncoder, KeySignatureEncoder
from tqdm.contrib.concurrent import process_map
from torch_geometric.data import InMemoryDataset, HeteroData
from analysisgnn.utils.dcl_tsv_utils import create_graph_from_df
import shutil
from typing import Union, List, Tuple


class AugmentedNetChordDataset(BuiltinDataset):
    r"""The AugmentedNet Chord Dataset.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if AugmentedNet Chord Dataset scores are already available otherwise it will download it.
        Default: ~/.struttura/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, is_zip=True, version="v1.0.0"):
        assert version in ["v1.0.0", "latest"]
        url = f"https://github.com/napulen/AugmentedNet/releases/download/{version}/dataset.zip"
        super(AugmentedNetChordDataset, self).__init__(
            name="AugmentedNetChordDataset",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)
        self.version = version

    def process(self, subset=""):
        self.scores = list()
        for root, dirs, files in os.walk(self.raw_path):
            if root.endswith(subset):
                for file in files:
                    if file.endswith(".tsv") and not file.startswith("dataset_summary"):
                        self.scores.append(os.path.join(root, file))

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True

        return False


class AugmentedNetv100Dataset(BuiltinDataset):
    """
    The AugmentedNet v1.0.0 Dataset compiled as individual notes in a tsv file.

    This dataset contains suplementary columns for each of the notes when comparing to the original AugmentedNet
    time separated dataset.
    """
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True):
        url = "https://github.com/johentsch/dilemmadata"
        super(AugmentedNetv100Dataset, self).__init__(
            name="dilemmadata",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            is_zip=False,
            clone=True,
            branch="main"            
        )
        self.scores = []
        self.collections = []
        self.composers = []
        self.period = "Unknown"
        self.type = []
        self.process()

    def process(self):
        self.scores = []
        self.collections = []
        self.composers = []
        folder_path = "pitch_arrays/AN/"
        path = os.path.join(self.raw_path, folder_path)
        assert os.path.isdir(path), "The directory does not exist maybe you need to checkout the correct branch"
        for split in ["training", "test", "validation"]:
            for file in os.listdir(os.path.join(path, split)):
                if file.endswith("joint.tsv"):
                    self.scores.append(os.path.join(path, split, file))
                    self.collections.append(split)

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True
        return False


class AugmentedNetLatestChordDataset(BuiltinDataset):
    r"""The AugmentedNet Chord Dataset.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if AugmentedNet Chord Dataset scores are already available otherwise it will download it.
        Default: ~/.struttura/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, is_zip=True):
        url = "https://github.com/napulen/AugmentedNet/releases/latest/download/dataset.zip"
        super(AugmentedNetLatestChordDataset, self).__init__(
            name="AugmentedNetLatestChordDataset",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def process(self, subset=""):
        self.scores = list()
        for root, dirs, files in os.walk(self.raw_path):
            if root.endswith(subset):
                for file in files:
                    if file.endswith(".tsv") and not file.startswith("dataset_summary"):
                        self.scores.append(os.path.join(root, file))

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True

        return False


class ChordGraphDataset(StrutturaDataset):
    def __init__(self, dataset_base, max_size=None, verbose=True, nprocs=1, name=None,
                 raw_dir=None, force_reload=False, prob_pieces=[], include_measures=False, transpose=False):
        self.dataset_base = dataset_base
        self.prob_pieces = prob_pieces
        self.transpose = transpose
        self.include_measures = include_measures
        self.dataset_base.process()
        self.max_size = max_size
        self.stage = "validate"
        if verbose:
            print("Loaded AugmentedNetChordDataset Successfully, now processing...")
        self.graph_dicts = list()
        self.n_jobs = nprocs
        super(ChordGraphDataset, self).__init__(
            name=name,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def process(self):
        Parallel(self.n_jobs)(delayed(self._process_score)(fn) for fn in
                              tqdm(self.dataset_base.scores, desc="Processing AugmentedNetChordGraphDataset"))
        self.load()

    def _process_score(self, score_fn):
        pass
    def has_cache(self):
        # return True
        if all([
            os.path.exists(os.path.join(self.save_path, os.path.splitext(os.path.basename(path))[0])) for path
            in
            self.dataset_base.scores]):
            return True
        return False

    def save(self):
        """save the graph list and the labels"""
        pass

    def load(self):
        self.graphs = list()
        for fn in os.listdir(self.save_path):
            path = os.path.join(self.save_path, fn)
            graph = load_score_hgraph(path, fn)
            if not self.include_synth and graph.name.endswith("-synth"):
                continue
            if self.collection != "all" and not graph.name.startswith(
                    self.collection) and graph.collection == "test":
                continue
            if graph.name in self.prob_pieces:
                continue
            self.graphs.append(graph)

    def set_split(self, stage):
        self.stage = stage

    @property
    def features(self):
        return self.graphs[0].x.shape[-1]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return [
            self.get_graph_attr(i)
            for i in idx
        ]

    def get_graph_attr(self, idx):
        out = dict()
        if self.graphs[idx].x.size(0) > self.max_size and self.graphs[idx].collection != "test":
            random_idx = random.randint(0, self.graphs[idx].x.size(0) - self.max_size)
            indices = torch.arange(random_idx, random_idx + self.max_size)
            edge_indices = torch.isin(self.graphs[idx].edge_index[0], indices) & torch.isin(
                self.graphs[idx].edge_index[1], indices)
            onset_divs = torch.tensor(
                self.graphs[idx].note_array["onset_div"][random_idx:random_idx + self.max_size])
            out["note_array"] = torch.vstack([torch.tensor(self.graphs[idx].note_array[el][random_idx:random_idx + self.max_size]) for el in ["pitch", "onset_beat", "duration_beat"]]).float().t()
            unique_onsets = torch.unique(torch.tensor(self.graphs[idx].note_array["onset_div"]), sorted=True)
            label_idx = (unique_onsets >= onset_divs.min()) & (unique_onsets <= onset_divs.max())
            out["x"] = self.graphs[idx].x[indices]
            out["edge_index"] = self.graphs[idx].edge_index[:, edge_indices] - random_idx
            out["edge_type"] = self.graphs[idx].edge_type[edge_indices]
            out["y"] = self.graphs[idx].y[label_idx]
            out["onset_div"] = onset_divs
            out["name"] = self.graphs[idx].name
            if self.include_measures:
                measure_edges = torch.tensor(self.graphs[idx].measure_edges)
                measure_nodes = torch.arange(len(self.graphs[idx].measure_nodes))
                beat_edges = torch.tensor(self.graphs[idx].beat_edges)
                beat_nodes = torch.arange(len(self.graphs[idx].beat_nodes))
                beat_edge_indices = torch.isin(beat_edges[0], indices)
                beat_node_indices = torch.isin(beat_nodes, torch.unique(beat_edges[1][beat_edge_indices]))
                min_beat_idx = torch.where(beat_node_indices)[0].min()
                max_beat_idx = torch.where(beat_node_indices)[0].max()
                measure_edge_indices = torch.isin(measure_edges[0], indices)
                measure_node_indices = torch.isin(measure_nodes, torch.unique(measure_edges[1][measure_edge_indices]))
                min_measure_idx = torch.where(measure_node_indices)[0].min()
                max_measure_idx = torch.where(measure_node_indices)[0].max()
                out["beat_nodes"] = beat_nodes[min_beat_idx:max_beat_idx+1] - min_beat_idx
                out["beat_edges"] = torch.vstack((beat_edges[0, beat_edge_indices] - random_idx,
                                                  beat_edges[1, beat_edge_indices] - min_beat_idx))
                out["measure_nodes"] = measure_nodes[min_measure_idx:max_measure_idx+1] - min_measure_idx
                out["measure_edges"] = torch.vstack((measure_edges[0, measure_edge_indices] - random_idx,
                                                     measure_edges[1, measure_edge_indices] - min_measure_idx))
        else:
            out["x"] = self.graphs[idx].x
            out["edge_index"] = self.graphs[idx].edge_index
            out["edge_type"] = self.graphs[idx].edge_type
            out["y"] = self.graphs[idx].y
            out["onset_div"] = torch.tensor(self.graphs[idx].note_array["onset_div"])
            out["note_array"] = torch.vstack(
                [torch.tensor(self.graphs[idx].note_array[el]) for el in
                 ["pitch", "onset_beat", "duration_beat"]]).t().float()
            out["name"] = self.graphs[idx].name
            if self.include_measures:
                out["beat_nodes"] = torch.tensor(self.graphs[idx].beat_nodes).squeeze()
                out["beat_edges"] = torch.tensor(self.graphs[idx].beat_edges)
                out["measure_nodes"] = torch.tensor(self.graphs[idx].measure_nodes).squeeze()
                out["measure_edges"] = torch.tensor(self.graphs[idx].measure_edges)
        return out


class AugmentedNetChordGraphDataset(ChordGraphDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, nprocs=4, include_synth=False, num_tasks=11,
                 collection="all", max_size=512, include_measures=False, transpose=False):
        dataset_base = AugmentedNetChordDataset(raw_dir=raw_dir)
        self.collection = collection
        # Collection is one of ["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern"]
        assert self.collection in ["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern", "all"]
        self.include_synth = include_synth
        # Problematic Pieces
        self.prob_pieces = []# ["bps-29-op106-hammerklavier-1", "tavern-mozart-k613-b", "tavern-mozart-k613-a", "abc-op127-4", "mps-k533-1", "abc-op59-no1-1"]
        # Frog model order: key, tonicisation, degree, quality, inversion, and root
        if isinstance(num_tasks, int):
            if num_tasks <= 6:
                self.tasks = {
                    "localkey": 35, "tonkey": 35, "degree1": 22, "degree2": 22, "quality": 16, "inversion": 4, "root": 35}
            elif num_tasks == 11:
                self.tasks = {
                    "localkey": 35, "tonkey": 35, "degree1": 22, "degree2": 22, "quality": 16, "inversion": 4,
                    "root": 35, "romanNumeral": 76, "hrhythm": 2, "pcset": 94, "bass": 35,
                }
        else:
            from analysisgnn.utils.chord_representations import available_representations
            self.tasks = {num_tasks: len(available_representations[num_tasks].classList)}
        super(AugmentedNetChordGraphDataset, self).__init__(
            dataset_base=dataset_base,
            max_size=max_size,
            nprocs=nprocs,
            name="AugmentedNetChordGraphDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            include_measures=include_measures,
            transpose=transpose
        )

    def _process_score(self, score_fn):
        name = os.path.splitext(os.path.basename(score_fn))[0]
        name = name + "-synth" if os.path.join("AugmentedNetChordDataset", "dataset-synth") in score_fn else name
        # name = name + "-2-synth" if os.path.join("AugmentedNetChordDataset", "dataset-synth-2") in score_fn else name
        # Skip synthetic scores in testing.

        if os.path.join("AugmentedNetChordDataset", "dataset-synth") in score_fn and os.path.basename(os.path.dirname(score_fn)) in ["test"]:
            return
        collection = "training" if os.path.basename(
            os.path.dirname(score_fn)) == "validation" else os.path.basename(os.path.dirname(score_fn))
        if collection == "test" or not self.transpose:
            note_array, labels, measures = time_divided_tsv_to_part(score_fn, transpose=False)
            data_to_graph(note_array, labels, collection, name, save_path=self.save_path, measures=measures)
        else:
            x = time_divided_tsv_to_part(score_fn, transpose=True)
            for i, (note_array, labels, measures) in enumerate(x):
                data_to_graph(note_array, labels, collection, (name + "-{}".format(i) if i > 0 else name), save_path=self.save_path, measures=measures)
        return


class Augmented2022ChordGraphDataset(ChordGraphDataset):

    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, nprocs=4, include_synth=False, num_tasks=11, collection="all",
                 max_size=512, include_measures=False, transpose=False):
        dataset_base = AugmentedNetLatestChordDataset(raw_dir=raw_dir)
        # Collection is one of ["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern"]
        self.collection = collection
        assert self.collection in ["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern", "mps", "all"]
        self.include_synth = include_synth
        # Problematic Pieces
        prob_pieces = [
            'keymodt-reger-96A',
            'keymodt-rimsky-korsakov-3-23c',
            'keymodt-reger-88A',
            'keymodt-reger-68',
            'keymodt-reger-84',
            'keymodt-rimsky-korsakov-3-24a',
            'keymodt-rimsky-korsakov-3-23b',
            'keymodt-aldwell-ex27-4b',
            'keymodt-reger-42a',
            'keymodt-reger-08',
            'mps-k545-1',
            'keymodt-tchaikovsky-173b',
            'keymodt-reger-73',
            'mps-k282-3',
            'keymodt-tchaikovsky-173j',
            'keymodt-kostka-payne-ex19-5',
            'mps-k457-3',
            'keymodt-reger-59',
            'keymodt-reger-82',
            'keymodt-rimsky-korsakov-3-5h',
            'mps-k332-2',
            'mps-k310-1',
            'mps-k457-2',
            'keymodt-rimsky-korsakov-3-7',
            'mps-k576-1',
            'keymodt-kostka-payne-ex18-4',
            'keymodt-reger-81',
            'keymodt-reger-45a',
            'keymodt-rimsky-korsakov-3-14g',
            'keymodt-reger-64',
            'keymodt-tchaikovsky-193b',
            'keymodt-reger-86A',
            'keymodt-reger-15',
            'keymodt-reger-28',
            'mps-k309-1',
            'keymodt-reger-99A',
            'keymodt-reger-55',
            'keymodt-tchaikovsky-189']

        # ["bps-29-op106-hammerklavier-1", "tavern-mozart-k613-b", "tavern-mozart-k613-a", "abc-op127-4", "mps-k533-1", "abc-op59-no1-1"]
        # Frog model order: key, tonicisation, degree, quality, inversion, and root
        if isinstance(num_tasks, int):
            if num_tasks <= 6:
                self.tasks = {
                    "localkey": 38, "tonkey": 38, "degree1": 22, "degree2": 22, "quality": 11, "inversion": 4,
                    "root": 35}
            elif num_tasks == 11:
                self.tasks = {
                    "localkey": 38, "tonkey": 38, "degree1": 22, "degree2": 22, "quality": 11, "inversion": 4,
                    "root": 35, "romanNumeral": 31, "hrhythm": 7, "pcset": 121, "bass": 35}
            elif num_tasks == 14:
                self.tasks = {
                    "localkey": 38, "tonkey": 38, "degree1": 22, "degree2": 22, "quality": 11, "inversion": 4,
                    "root": 35, "romanNumeral": 31, "hrhythm": 7, "pcset": 121, "bass": 35, "tenor": 35,
                    "alto": 35, "soprano": 35}
        else:
            from analysisgnn.utils.chord_representations_latest import available_representations
            self.tasks = {num_tasks: len(available_representations[num_tasks].classList)}
        super(Augmented2022ChordGraphDataset, self).__init__(
            dataset_base=dataset_base,
            max_size=max_size,
            nprocs=nprocs,
            name="Augmented2022ChordGraphDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            prob_pieces=prob_pieces,
            include_measures=include_measures,
            transpose=transpose
        )

    def _process_score(self, score_fn):
        name = os.path.splitext(os.path.basename(score_fn))[0]
        name = name + "-synth" if os.path.join("AugmentedNetLatestChordDataset", "dataset-synth") in score_fn else name
        # Skip synthetic scores in testing.
        if os.path.join("AugmentedNetLatestChordDataset", "dataset-synth") in score_fn and os.path.basename(
                os.path.dirname(score_fn)) in ["test"]:
            return
        collection = "training" if os.path.basename(
            os.path.dirname(score_fn)) == "validation" else os.path.basename(os.path.dirname(score_fn))
        if collection == "test" or not self.transpose:
            note_array, labels = time_divided_tsv_to_part(score_fn, transpose=False, version="latest")
            data_to_graph(note_array, labels, collection, name, save_path=self.save_path)
        else:
            x = time_divided_tsv_to_part(score_fn, transpose=True, version="latest")
            for i, (note_array, labels) in enumerate(x):
                data_to_graph(note_array, labels, collection, (name + "-{}".format(i) if i > 0 else name),
                              save_path=self.save_path)
        return


def data_to_graph(note_array, labels, collection, name, save_path, measures=None):
    nodes, edges = hetero_graph_from_note_array(note_array=note_array)
    note_features = select_features(note_array, "chord")
    hg = HeteroScoreGraph(
        note_features,
        edges,
        name=name,
        labels=labels,
        note_array=note_array,
    )
    setattr(hg, "collection", collection)
    if measures is not None:
        hg.add_beat_nodes()
        hg.add_measure_nodes(measures)
    # pos_enc = positional_encoding(hg.edge_index, len(hg.x), 20)
    # hg.x = torch.cat((hg.x, pos_enc), dim=1)
    hg.save(save_path)
    del hg, note_array, nodes, edges, note_features
    return


class TestRNAGraphDataset(ChordGraphDataset):

    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, nprocs=4, include_synth=False, num_tasks=11, collection="all",
                 max_size=512, include_measures=False, transpose=False):
        dataset_base = AugmentedNetChordDataset(raw_dir=raw_dir, version="v1.0.0")
        # Collection is one of ["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern"]
        self.collection = collection
        assert self.collection in ["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern", "mps", "all"]
        self.include_synth = include_synth
        # Problematic Pieces
        prob_pieces = [
            'keymodt-reger-96A',
            'keymodt-rimsky-korsakov-3-23c',
            'keymodt-reger-88A',
            'keymodt-reger-68',
            'keymodt-reger-84',
            'keymodt-rimsky-korsakov-3-24a',
            'keymodt-rimsky-korsakov-3-23b',
            'keymodt-aldwell-ex27-4b',
            'keymodt-reger-42a',
            'keymodt-reger-08',
            'mps-k545-1',
            'keymodt-tchaikovsky-173b',
            'keymodt-reger-73',
            'mps-k282-3',
            'keymodt-tchaikovsky-173j',
            'keymodt-kostka-payne-ex19-5',
            'mps-k457-3',
            'keymodt-reger-59',
            'keymodt-reger-82',
            'keymodt-rimsky-korsakov-3-5h',
            'mps-k332-2',
            'mps-k310-1',
            'mps-k457-2',
            'keymodt-rimsky-korsakov-3-7',
            'mps-k576-1',
            'keymodt-kostka-payne-ex18-4',
            'keymodt-reger-81',
            'keymodt-reger-45a',
            'keymodt-rimsky-korsakov-3-14g',
            'keymodt-reger-64',
            'keymodt-tchaikovsky-193b',
            'keymodt-reger-86A',
            'keymodt-reger-15',
            'keymodt-reger-28',
            'mps-k309-1',
            'keymodt-reger-99A',
            'keymodt-reger-55',
            'keymodt-tchaikovsky-189']

        # ["bps-29-op106-hammerklavier-1", "tavern-mozart-k613-b", "tavern-mozart-k613-a", "abc-op127-4", "mps-k533-1", "abc-op59-no1-1"]
        # Frog model order: key, tonicisation, degree, quality, inversion, and root
        if isinstance(num_tasks, int):
            if num_tasks <= 6:
                self.tasks = {
                    "localkey": 38, "tonkey": 38, "degree1": 22, "degree2": 22, "quality": 11, "inversion": 4,
                    "root": 35}
            elif num_tasks == 11:
                self.tasks = {
                    "localkey": 38, "tonkey": 38, "degree1": 22, "degree2": 22, "quality": 11, "inversion": 4,
                    "root": 35, "romanNumeral": 31, "hrhythm": 7, "pcset": 121, "bass": 35}
            elif num_tasks == 14:
                self.tasks = {
                    "localkey": 38, "tonkey": 38, "degree1": 22, "degree2": 22, "quality": 11, "inversion": 4,
                    "root": 35, "romanNumeral": 31, "hrhythm": 7, "pcset": 121, "bass": 35, "tenor": 35,
                    "alto": 35, "soprano": 35}
        else:
            from analysisgnn.utils.chord_representations_latest import available_representations
            self.tasks = {num_tasks: len(available_representations[num_tasks].classList)}
        self.pitch_encoder = PitchEncoder()
        super(TestRNAGraphDataset, self).__init__(
            dataset_base=dataset_base,
            max_size=max_size,
            nprocs=nprocs,
            name="TestRNAGraphDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            prob_pieces=prob_pieces,
            include_measures=include_measures,
            transpose=transpose
        )

    def _process_score(self, score_fn):
        name = os.path.splitext(os.path.basename(score_fn))[0]
        name = name + "-synth" if os.path.join("AugmentedNetLatestChordDataset", "dataset-synth") in score_fn else name
        # Skip synthetic scores in testing.
        if os.path.join("AugmentedNetLatestChordDataset", "dataset-synth") in score_fn and os.path.basename(
                os.path.dirname(score_fn)) in ["test"]:
            return
        collection = "training" if os.path.basename(
            os.path.dirname(score_fn)) == "validation" else os.path.basename(os.path.dirname(score_fn))
        if os.path.exists(os.path.join(self.raw_path, name + ".pt")):
            return torch.load(os.path.join(self.raw_path, name + ".pt"))
        if collection == "test" or not self.transpose:
            note_array, labels, measures, label_names = time_divided_tsv_to_part(score_fn, transpose=False, version=self.dataset_base.version)
            graph = self.data_to_graph(note_array, labels, measures=measures, label_names=label_names)
            graph.name = name
            graph.collection = collection
            torch.save(graph, os.path.join(self.raw_path, name+".pt"))
        else:
            x = time_divided_tsv_to_part(score_fn, transpose=True, version=self.dataset_base.version)
            raise NotImplementedError("Transpose not implemented for TestRNAGraphDataset")
            for i, (note_array, labels, measures, label_names) in enumerate(x):
                graph = self.data_to_graph(note_array, labels, measures=measures, label_names=label_names)
                graph.name = name
                graph.collection = collection
                torch.save(graph, os.path.join(self.raw_path, name))
        return graph

    def data_to_graph(self, note_array, labels, measures=None, label_names=[]):
        assert np.all(note_array["onset_div"] == sorted(note_array["onset_div"]))
        unique_onset, inverse_idxs = torch.unique(torch.tensor(note_array["onset_div"]), sorted=True,
                                                  return_inverse=True)
        assert len(unique_onset) == len(labels)
        # produce labels for each note in the score (previously only for unique onsets)
        labels = labels[inverse_idxs]

        features = select_features(note_array, "voice")
        graph = create_score_graph(features, note_array=note_array, add_beats=True, measures=measures)
        for i, label in enumerate(label_names):
            graph["note"][label] = torch.tensor(labels[:, i], dtype=torch.long).squeeze()
        # encode pitch_spelling
        pitch_spelling = self.pitch_encoder.encode(note_array)
        graph["note"].pitch_spelling = torch.tensor(pitch_spelling, dtype=torch.long).squeeze()
        return graph

    def process(self):
        self.graphs = process_map(self._process_score, self.dataset_base.scores, desc="Processing RNAGraphDataset")

    def has_cache(self):
        # return True
        if all([
            os.path.exists(os.path.join(self.save_path, os.path.splitext(os.path.basename(path))[0]) + ".pt") for path
            in
            self.dataset_base.scores]):
            return True
        return False


class RNAGraphDataset(InMemoryDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, num_workers=1, transpose=False, name=None):
        name = "RNAGraphDataset" if name is None else name
        self.base_dataset = AugmentedNetv100Dataset(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)
        root = os.path.join(get_download_dir(), name) if raw_dir is None else os.path.join(
            raw_dir, name)
        self.verbose = verbose
        self.all_intervals = ["P1", "m2", "M2", "m3", "M3", "P4", "A4", "P5", "m6", "M6", "m7", "M7"]
        self.transpose = transpose
        self.force_reload = force_reload
        self.num_workers = num_workers
        transform, pre_filter, pre_transform = None, None, None
        # NOTE: Add problematic pieces here to avoid processing them.
        self.prob_pieces = [
            # "wir-openscore-liedercorpus-schubert-winterreise-d-911-06-wasserfluth_joint",

        ]
        # NOTE: Add test pieces to predefine the test split.
        self.test_pieces = []
        super(RNAGraphDataset, self).__init__(root, transform, pre_filter)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        fns = []
        if not os.path.exists(self.raw_dir) and not os.path.isdir(self.raw_dir):
            os.makedirs(self.raw_dir)
        for ofp in self.base_dataset.scores:
            fn = os.path.basename(ofp)
            if fn in self.prob_pieces:
                continue
            elif not os.path.exists(os.path.join(self.raw_dir, fn)):
                shutil.copy(ofp, os.path.join(self.raw_dir, fn))
                fns.append(fn)
            else:
                fns.append(fn)

        return fns

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ["data.pt"]

    def _process_single(self, data):
        raw_path, collection, spec_file, converters, pitch_encoder, key_signature_encoder = data
        data_list = []
        if os.path.splitext(os.path.basename(raw_path))[0] in self.prob_pieces:
            return data_list

        split = collection
        intervals = self.all_intervals if self.transpose and collection == "training" else ["P1"]
        gname = os.path.splitext(os.path.basename(raw_path))[0]
        for interval in intervals:
            name = f"{gname}_{interval}" if interval != "P1" else gname
            save_fp = os.path.join(self.processed_dir, name + ".pt")

            if os.path.exists(save_fp) and not self.force_reload:
                d = torch.load(save_fp)
                data_list.append(d)
                continue

            try:
                d = create_graph_from_df(raw_path, pitch_encoder=pitch_encoder, do_labels=True, drop_na=True,
                                         dlc=False, spec_file=spec_file, converters=converters, interval=interval,
                                         key_signature_encoder=key_signature_encoder)
                d["split"] = split
                d["name"] = gname
                d["interval"] = interval
                d["test"] = collection == "test"
                torch.save(d, save_fp)

            except Exception as e:
                d = None
                if self.verbose:
                    print("Error processing {} with transposition {}".format(raw_path, interval))

            # except:
            #     print("Error processing {}".format(raw_path))
            #     continue
            if d is not None:
                data_list.append(d)
        return data_list

    def process(self):
        from analysisgnn.models.pitch_spelling import PitchEncoder
        from analysisgnn.utils.dcl_tsv_utils import create_graph_from_df, create_spec_file
        from tqdm import tqdm
        from tqdm.contrib.concurrent import process_map

        data_list = []
        pitch_encoder = PitchEncoder()
        key_signature_encoder = KeySignatureEncoder()
        # replace_dtypes = dict(object="string", int64="Int64")
        spec_file_path = os.path.join(self.base_dataset.raw_path, "processing", "DLC", "dlc_pitch_array_specs.csv")
        # "https://raw.githubusercontent.com/johentsch/AugmentedNet/refs/heads/v100_notes/augnet_pitch_array_specs.csv"
        assert os.path.exists(spec_file_path), "The spec file does not exist, or is not in the correct location."
        spec_file, converters = create_spec_file(spec_file_path)
        if self.num_workers > 1:
            data_list = process_map(self._process_single, [(raw_path, collection, spec_file, converters, pitch_encoder, key_signature_encoder) for collection, raw_path in zip(self.base_dataset.collections, self.base_dataset.scores)], desc="Processing RNAGraphDataset", max_workers=self.num_workers)
            data_list = [d for dl in data_list for d in dl]
        else:
            for collection, raw_path in tqdm(zip(self.base_dataset.collections, self.base_dataset.scores), desc="Processing RNAGraphDataset"):
                data = (raw_path, collection, spec_file, converters, pitch_encoder, key_signature_encoder)
                d = self._process_single(data)
                data_list.extend(d)



        self.save(data_list, self.processed_paths[0])



class RNAplusGraphDataset(RNAGraphDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, num_workers=1, transpose=False):
        name = "RNAplusGraphDataset"
        super(RNAplusGraphDataset, self).__init__(
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            num_workers=num_workers,
            transpose=transpose,
            name=name
        )

    def _process_single(self, data):
        raw_path, collection, spec_file, converters, pitch_encoder, key_signature_encoder = data
        data_list = []
        if os.path.splitext(os.path.basename(raw_path))[0] in self.prob_pieces:
            return data_list

        split = collection
        intervals = self.all_intervals if self.transpose and collection == "training" else ["P1"]
        gname = os.path.splitext(os.path.basename(raw_path))[0]
        for interval in intervals:
            name = f"{gname}_{interval}" if interval != "P1" else gname
            save_fp = os.path.join(self.processed_dir, name + ".pt")

            if os.path.exists(save_fp) and not self.force_reload:
                d = torch.load(save_fp)
                data_list.append(d)
                continue

            try:
                d = create_graph_from_df(raw_path, pitch_encoder=pitch_encoder, do_labels=True, drop_na=True,
                                         dlc=False, spec_file=spec_file, converters=converters, interval=interval,
                                         key_signature_encoder=key_signature_encoder, feature_type="cadence")
                d["split"] = split
                d["name"] = gname
                d["interval"] = interval
                d["transposition"] = interval
                d["test"] = collection == "test"
                torch.save(d, os.path.join(self.processed_dir, name))

            except Exception as e:
                d = None
                if self.verbose:
                    print("Error processing {} with transposition {}".format(raw_path, interval))

            # except:
            #     print("Error processing {}".format(raw_path))
            #     continue
            if d is not None:
                data_list.append(d)
        return data_list
