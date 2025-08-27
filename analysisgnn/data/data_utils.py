import os
import torch
import warnings
import partitura as pt
import numpy as np
import shutil
from typing import List, Tuple, Union
from analysisgnn.utils.globals import INTERVAL_TO_SEMITONES
from analysisgnn.descriptors import select_features
from graphmuse import create_score_graph
from analysisgnn.utils.music import PitchEncoder, KeySignatureEncoder
from collections import defaultdict
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, HeteroData
from analysisgnn.data.dataset import get_download_dir


class StandardGraphDataset(InMemoryDataset):
    """
    StandardGraphDataset is a dataset class that extends PyTorch Geometric's InMemoryDataset.
    It processes a base dataset into a graph representation suitable for graph neural networks.

    Attributes:
        base_dataset: The base dataset to be processed.
        raw_dir: Directory where raw data is stored, if None, points to ~/.struttura.
        force_reload: If True, forces reloading of the dataset even if it exists.
        verbose: If True, enables verbose logging.
        num_workers: Number of worker processes to use for data loading. Default is 1. If use_mp is True, this is ignored.
        prob_pieces: List of problematic pieces to skip during processing. Default is an empty list.
        use_mp: If True, uses multiprocessing for data processing. Default is True. Set to False in case of issues.
        skip_reload: If True, skips reloading the dataset if it exists. Useful for big datasets or debugging.
        enumerate_files: If True, enumerates raw files during processing. The enumeration is used to name the processed files to avoid overwriting.
    """
    def __init__(self, base_dataset, raw_dir=None, force_reload=False, verbose=True,
                 num_workers=1, prob_pieces=[], use_mp=True, skip_reload=False, enumerate_files=True):
        self.base_dataset = base_dataset
        self.base_dataset.process()
        name = self.base_dataset.name.replace("Dataset", "") + "GraphDataset"
        root = os.path.join(get_download_dir(), name) if raw_dir is None else os.path.join(
         raw_dir, "name")
        self.verbose = verbose
        self.force_reload = force_reload
        self.num_workers = num_workers
        self.skip_reload = skip_reload
        self.test_pieces = []
        self.not_parsed = []
        self.enumerate_files = enumerate_files
        self.use_mp = use_mp
        transform, pre_filter, pre_transform = None, None, None
        self.prob_pieces = prob_pieces
        super(StandardGraphDataset, self).__init__(root, transform, pre_filter)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        fns = []
        if not os.path.exists(self.raw_dir) and not os.path.isdir(self.raw_dir):
            os.makedirs(self.raw_dir)
        if self.skip_reload:
            return list(map(lambda x: os.path.join(self.raw_dir, x), os.listdir(self.raw_dir)))
        for i, ofp in enumerate(self.base_dataset.scores):
            fn = ofp.replace(self.base_dataset.raw_path, "")
            nfn = str(i) + "_" + os.path.basename(ofp) if self.enumerate_files else os.path.basename(ofp)
            if fn in self.prob_pieces:
                if self.verbose:
                    print(f"Skipping {fn}")
                continue
            elif not os.path.exists(os.path.join(self.raw_dir, nfn)):
                shutil.copy(ofp, os.path.join(self.raw_dir, nfn))
                fns.append(nfn)
            else:
                fns.append(nfn)

        return fns

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ["data.pt"]

    def process_loop(self, raw_path):
        save_fp = os.path.join(self.processed_dir, os.path.splitext(os.path.basename(raw_path))[0] + ".pt")
        if os.path.exists(save_fp) and not self.force_reload:
            d = torch.load(save_fp)
            return d
        if os.path.splitext(os.path.basename(raw_path))[0] in self.prob_pieces:
            return None
        try:
            d = process_score_pitch_spelling({"name": os.path.splitext(os.path.basename(raw_path))[0],
                                          "path": raw_path, "save_path": self.processed_dir,
                                          "force_reload": self.force_reload, "verbose": self.verbose}
            )
            return d
        except Exception as e:
            if self.verbose:
               print(e)
            print(f"Error processing {raw_path}")
            self.not_parsed.append(raw_path)
            return None

    def process(self):
        if self.use_mp:
            data_list = process_map(self.process_loop, self.raw_paths, max_workers=self.num_workers)
            data_list = [d for d in data_list if d is not None]
        else:
            data_list = []
            for raw_path in tqdm(self.raw_paths, desc="Processing scores to graphs"):
                d = self.process_loop(raw_path)
                if d is not None:
                    data_list.append(d)

        if self.verbose:
            print(f"Could not parse {len(self.not_parsed)} pieces: \n {self.not_parsed}")

        self.save(data_list, self.processed_paths[0])

    @property
    def num_features(self) -> int:
        return self[0]["note"].num_features


class SimpleGraphDataset(InMemoryDataset):
    """
    Used for datasets that have already been processed to torch_geometric data objects
    Transforms a list of torch_geometric data objects to a PyTorch Geometric InMemoryDataset
    """
    def __init__(self, data_list, name, root=None, transform=None, pre_transform=None):
        self.data_list = data_list
        self.name = name
        self.data, self.slices = self.collate(self.data_list)
        super(SimpleGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ["data.pt"]

    def process(self):
        torch.save((self.data, self.slices), self.processed_paths[0])

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def process_score_pitch_spelling(info, save=True, allow_monophonic=False, interval="P1", feature_type="voice"):
    """
    Processes a score to a graph representation suitable for graph neural networks.

    Args:
        info: A dictionary containing the following keys:
            name: The name of the piece.
            path: The path to the piece.
            save_path: The path to save the processed graph.
            force_reload: If True, forces reloading of the dataset even if it exists. Default is True.
            verbose: If True, enables verbose logging. Default is False.
            prob_pieces: List of problematic pieces to skip during processing. Default is an empty list.
        save: If True, saves the processed graph. Default is True.
        allow_monophonic: If True, allows monophonic pieces to be processed. Default is False.
    """
    name = info["name"]
    score_fn = info["path"]
    save_path = info["save_path"]
    force_reload = info.get("force_reload", True)
    verbose = info.get("verbose", False)
    label_func = info.get("label_func", None)
    prob_pieces = info.get("prob_pieces", [])
    save_path = os.path.join(save_path, name + ".pt")

    if name in prob_pieces:
        return
    if verbose:
        print("Processing {}".format(name))
        score = pt.load_kern(score_fn, force_same_part=True) if score_fn.endswith(".krn") else pt.load_score(score_fn)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = pt.load_kern(score_fn, force_same_part=True) if score_fn.endswith(".krn") else pt.load_score(score_fn)
    measures = score[-1].measures
    part = score[-1]
    note_array = score.note_array(include_time_signature=True, include_pitch_spelling=True,
                                  include_key_signature=True, include_staff=True, include_metrical_position=True)
    note_array = np.sort(note_array, order=["onset_div", "pitch"])
    note_array["pitch"] = (note_array["pitch"] + INTERVAL_TO_SEMITONES[interval]) % 128

    note_features = select_features(note_array, feature_type)

    labels = label_func(note_array, score) if label_func is not None else None
    data = create_score_graph(note_features, note_array, measures=measures, add_beats=True, labels=labels)

    # Skip monophonic pieces
    if not allow_monophonic:
        if torch.all(data["note", "onset", "note"].edge_index[0] == data["note", "onset", "note"].edge_index[1]) and \
                data["note", "during", "note"].edge_index.shape[1] == 0:
            return None
    pitch_encoder = PitchEncoder()
    ks_encoder = KeySignatureEncoder()
    labels_ps = pitch_encoder.encode(note_array)
    labels_ps = pitch_encoder.transpose(labels_ps, interval)
    labels_ks = ks_encoder.encode(note_array)
    labels_ks = ks_encoder.transpose(labels_ks, interval)
    data["note"].pitch_spelling = torch.from_numpy(labels_ps).long()
    data["note"].key_signature = torch.from_numpy(labels_ks).long()
    data["note"].voice = torch.from_numpy(note_array["voice"]).long()
    data["note"].staff = torch.from_numpy(note_array["staff"]).long()
    # asign name to the graph
    data["name"] = name
    data["transposition"] = interval
    if save:
        torch.save(data, save_path)
    return data


class CummulativeDataset(InMemoryDataset):
    def __init__(self, name, datasets, transform=None, raw_dir=None):
        self.data_list = []
        datasets = datasets
        if raw_dir is None:
            droots = [os.path.dirname(d.root) for d in datasets]
            root = os.path.join(get_download_dir(), name) if len(set(droots)) != 1 else os.path.join(droots[0], name)
        else:
            root = os.path.join(raw_dir, name)
        self.name = name
        self.setup_datasets(datasets, root)
        super().__init__(root, transform)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def setup_datasets(self, datasets, root):
        if not os.path.exists(os.path.join(root, "processed", "data.pt")):
            for d in datasets:
                for idx in range(d.len()):
                    self.data_list.append(d[idx])

    def process(self):
        if self.transform is not None:
            data_list = []
            fail_names = []
            failed = 0
            for data in self.data_list:
                try:
                    data = self.transform(data)
                except:
                    failed += 1
                    fail_names.append(data["note"].name)
                    continue
                data_list.append(data)
            self.data_list = data_list
            print(f"Failed to process {failed} graphs")
            print(f"Failed names: {fail_names}")
        self.save(self.data_list, self.processed_paths[0])


def idx_tuple_to_dict(idx_tuple, datasets_map):
    """Transforms indices of a list of tuples of indices (dataset, piece_in_dataset)
    into a dict {dataset: [piece_in_dataset,...,piece_in_dataset]}"""
    result_dict = defaultdict(list)
    for x in idx_tuple:
        result_dict[datasets_map[x][0]].append(datasets_map[x][1])
    return result_dict

def idx_dict_to_tuple(idx_dict):
    result_tuples = list()
    for k in idx_dict.keys():
        for v in idx_dict[k]:
            result_tuples.append((k,v))
    return result_tuples


def struttura_to_inmemory_dataset(struttura_dataset):
    d = struttura_dataset
    data_list = d.graphs
    raw_dir = d.raw_dir
    name = d.name
    root = os.path.join(get_download_dir(), name) if raw_dir is None else os.path.join(raw_dir, name)
    return SimpleGraphDataset(data_list=data_list, name=name, root=root)
