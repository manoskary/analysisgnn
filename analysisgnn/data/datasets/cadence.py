from typing import Union, List, Tuple
from torch_geometric.data import InMemoryDataset, HeteroData
from analysisgnn.data import BuiltinDataset
from analysisgnn.data.dataset import StrutturaDataset, get_download_dir
from analysisgnn.utils.music import CadenceEncoder
from analysisgnn.data.data_utils import process_score_pitch_spelling
from analysisgnn.descriptors import cadence_features
from graphmuse import create_score_graph
from analysisgnn.utils import partitura as pt
from tqdm.contrib.concurrent import process_map
import os
import torch
import numpy as np
import warnings
from analysisgnn.utils.music import transpose_note_array


class XMLCadenceDataset(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, branch="cadence_no_dcml"):
        url = "https://github.com/manoskary/cadence_xml_datasets"
        self.scores = list()
        super(XMLCadenceDataset, self).__init__(
            name="XMLCadenceDataset",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            clone=True,
            branch=branch
        )

    def process(self):
        for root, _, files in os.walk(os.path.join(self.raw_path, "data")):
            for file in files:
                if file.endswith(".musicxml"):
                    self.scores.append(os.path.join(root, file))


class CadenceGraphDataset(StrutturaDataset):
    def __init__(self, dataset_base, raw_dir=None, verbose=True, num_workers=4, force_reload=False, **kwargs):
        self.num_workers = num_workers
        self.dataset_base = dataset_base
        self.graphs = list()
        self.test_pieces = kwargs.get("test_pieces", [])
        self.prob_pieces = kwargs.get("prob_pieces", [])
        self.cadence_encoder = CadenceEncoder()
        self.process_func = kwargs.get("process_func", None)
        if verbose:
            print("Loaded Dependency Repository for Cadences Successfully, now processing...")
        super(CadenceGraphDataset, self).__init__(
            name=dataset_base.name.replace("CadenceDataset", "GraphCadenceDataset"),
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        scores = self.dataset_base.scores
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # process the scores
        self.graphs = process_map(self.process_score, scores, max_workers=self.num_workers)

    def augment(self, ratio: int = 3):
        if not os.path.exists(os.path.join(self.raw_path, "augmentations")):
            os.makedirs(os.path.join(self.raw_path, "augmentations"))
            if self.verbose:
                print("Augmenting the dataset with transpositions...")
            names = [os.path.splitext(os.path.basename(score))[0] for score in self.dataset_base.scores]
            scores = list(zip(self.dataset_base.scores, names))
            scores = [(path, name) for path, name in scores if name not in self.prob_pieces and name not in self.test_pieces]
            aug_graphs = list()
            for _ in range(ratio):
                ag = process_map(
                    self._process_score_aug, scores, max_workers=self.num_workers)
                aug_graphs.extend(ag)
            # filter out the nan values
            aug_graphs = [g for g in aug_graphs if g is not None]
            self.graphs.extend(aug_graphs)
        else:
            self.load_augmented()

    def _process_score_aug(self, info):
        score_fn, name = info
        intervals = [(2, "m"), (2, "M"), (3, "m"), (3, "M"), (4, "P"), (5, "P"), (6, "m"), (6, "M"), (7, "m"),
                     (7, "M")]
        random_permutation = np.random.permutation(intervals)

        if self.process_func is not None:
            info = {
                "name": name, "path": score_fn, "save_path": self.save_path, "verbose": self.verbose,
                "force_reload": self._force_reload, "label_func": self.produce_labels}
            for interval in random_permutation:
                try:
                    return self.process_func(info, interval=interval[1]+str(interval[0]))
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing {name} with interval {interval}: {e}")
                    continue

        if self.verbose:
            print("Processing {}".format(name))
            score = pt.load_score(score_fn)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = pt.load_score(score_fn)
        measures = score[-1].measures
        cadences = score[-1].cadences
        if cadences == []:
            return
        note_array = score.note_array(include_time_signature=True, include_pitch_spelling=True,
                                      include_key_signature=True, include_metrical_position=True)
        random_permutation = np.random.permutation(intervals)

        for interval in random_permutation:
            save_path = os.path.join(self.save_path, "augmentations",
                                     name + "_" + str(interval[0]) + interval[1] + ".pt")
            if os.path.exists(save_path):
                continue
            try:
                transposed_note_array = transpose_note_array(note_array, pt.score.Interval(number=eval(interval[0]), quality=interval[1]))
                labels = self.cadence_encoder.encode(transposed_note_array, cadences)
                features, _ = cadence_features(transposed_note_array)
                graph = create_score_graph(features, transposed_note_array, labels=labels, add_beats=True, measures=measures)
                graph.name = name
                torch.save(graph, save_path)
                return graph
            except:
                pass

    def process_score(self, score):
        name = os.path.splitext(os.path.basename(score))[0]
        if self.process_func is not None:
            info = {
                "name": name, "path": score, "save_path": self.save_path, "verbose": self.verbose,
                "force_reload": self._force_reload, "label_func": self.produce_labels}
            return self.process_func(info)
        else:
            save_graph_path = os.path.join(self.save_path, name + ".pt")
            if self._force_reload or not os.path.exists(save_graph_path):
                # combine name with dirname
                if self.verbose:
                    print(f"Processing {name}")
                score = pt.load_score(score)
                note_array = score.note_array(include_time_signature=True, include_metrical_position=True, include_pitch_spelling=True)
                part = score.parts[-1]
                measures = part.measures
                labels = self.produce_labels(note_array, score)
                features, _ = cadence_features(note_array)
                graph = create_score_graph(features, note_array, labels=labels, add_beats=True, measures=measures)
                graph.name = name
                # save graph to disk (i.e. a torch_geometric data object)
                torch.save(graph, save_graph_path)
                # return the graph
                return graph
            return torch.load(save_graph_path)

    def produce_labels(self, note_array, score):
        cadences = score[-1].cadences
        return self.cadence_encoder.encode(note_array, cadences)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        self.graphs = list()
        for fn in os.listdir(self.save_path):
            path = os.path.join(self.save_path, fn)
            graph = torch.load(path)
            self.graphs.append(graph)

    def load_augmented(self):
        for data in os.listdir(os.path.join(self.save_path, "augmentations")):
            if data.endswith(".pt"):
                self.graphs.append(torch.load(os.path.join(self.save_path, "augmentations", data)))

    def has_cache(self):
        scores = [os.path.splitext(os.path.basename(score))[0] for score in self.dataset_base.scores]
        if np.all([os.path.exists(os.path.join(self.save_path, os.path.basename(score) + ".pt")) for score in scores]):
            return True

        return False

    @property
    def save_name(self):
        return self.name

    @property
    def features(self):
        return self.graphs[0]["note"].x.shape[-1]


class CompleteGraphCadenceDataset(CadenceGraphDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, num_workers=4,
                 process_func=None):
        dataset_base = XMLCadenceDataset(
            raw_dir=raw_dir, force_reload=force_reload, verbose=verbose, branch="cadence_no_dcml"
        )
        prob_pieces = []
        super(CompleteGraphCadenceDataset, self).__init__(
            dataset_base=dataset_base,
            raw_dir=raw_dir,
            verbose=verbose,
            num_workers=num_workers,
            force_reload=force_reload,
            prob_pieces=prob_pieces,
            process_func=process_func
        )

def cadence_label_func(note_array, score):
    cadence_encoder = CadenceEncoder()
    cadences = score[-1].cadences
    labels = cadence_encoder.encode(note_array, cadences)
    return labels


class CadenceGraphPGDataset(InMemoryDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, num_workers=1, transpose=False, name=None):
        name = "CadenceGraphPGDataset" if name is None else name
        self.base_dataset = XMLCadenceDataset(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)
        root = os.path.join(get_download_dir(), name) if raw_dir is None else os.path.join(
            raw_dir, name)
        self.verbose = verbose
        self.force_reload = force_reload
        self.num_workers = num_workers
        self.transpose = transpose
        self.all_intervals = ["P1", "m2", "M2", "m3", "M3", "P4", "A4", "P5", "m6", "M6", "m7", "M7"]
        transform, pre_filter, pre_transform = None, None, None
        # NOTE: Add problematic pieces here to avoid processing them.
        self.prob_pieces = []
        # NOTE: Add test pieces to predefine the test split.
        self.test_pieces = []
        super(CadenceGraphPGDataset, self).__init__(root, transform, pre_filter)
        self.load(self.processed_paths[0], data_cls=HeteroData)


    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        pass

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ["data.pt"]

    def _process_single(self, raw_path):
        data_list = []
        gname = os.path.splitext(os.path.basename(raw_path))[0]
        intervals = self.all_intervals if gname not in self.test_pieces and self.transpose else ["P1"]
        for interval in intervals:
            name = f"{gname}_{interval}" if interval != "P1" else gname
            save_fp = os.path.join(self.processed_dir, f"{name}.pt")
            if os.path.exists(save_fp) and not self.force_reload:
                d = torch.load(save_fp)
                data_list.append(d)
                continue
            if os.path.splitext(os.path.basename(raw_path))[0] in self.prob_pieces:
                continue

            info = {
                "name": name,
                "path": raw_path,
                "save_path": self.processed_dir,
                "verbose": self.verbose,
                "force_reload": self.force_reload,
                "label_func": cadence_label_func
            }
            try:
                d = process_score_pitch_spelling(save=False, info=info, feature_type="voice", interval=interval)
                d["note"].cadence = d["note"].y
                d["test"] = gname in self.test_pieces
                torch.save(d, save_fp)
            except:
                if self.verbose:
                    print("Error processing {}".format(raw_path))
            # except:
            #     print("Error processing {}".format(raw_path))
            #     continue
            if d is not None:
                data_list.append(d)
        return data_list

    def process(self):
        from tqdm import tqdm
        from tqdm.contrib.concurrent import process_map

        if self.test_pieces == []:
            # perform random split to select 20% of the pieces as test pieces
            test_pieces = np.random.choice(self.base_dataset.scores, int(0.2*len(self.base_dataset.scores)), replace=False)
            self.test_pieces = [os.path.splitext(os.path.basename(piece))[0] for piece in test_pieces]

        data_list = []
        if self.num_workers > 1:
            data_list = process_map(self._process_single, self.base_dataset.scores, max_workers=self.num_workers)
            data_list = [d for dl in data_list for d in dl]
        else:
            for raw_path in tqdm(self.base_dataset.scores, desc="Processing CadenceGraphDataset"):
                d = self._process_single(raw_path)
                data_list.extend(d)

        self.save(data_list, self.processed_paths[0])


class CadenceSimpleGraphPGDataset(CadenceGraphPGDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, num_workers=1, transpose=False):
        super(CadenceSimpleGraphPGDataset, self).__init__(
            raw_dir=raw_dir, force_reload=force_reload,
            verbose=verbose, num_workers=num_workers, transpose=transpose, name="CadenceSimpleGraphPGDataset")

    def _process_single(self, raw_path):
        data_list = []

        gname = os.path.splitext(os.path.basename(raw_path))[0]
        intervals = self.all_intervals if gname not in self.test_pieces and self.transpose else ["P1"]
        for interval in intervals:
            name = f"{gname}_{interval}" if interval != "P1" else gname
            save_fp = os.path.join(self.processed_dir, f"{name}.pt")
            if os.path.exists(save_fp) and not self.force_reload:
                d = torch.load(save_fp)
                data_list.append(d)
                continue
            if os.path.splitext(os.path.basename(raw_path))[0] in self.prob_pieces:
                continue

            info = {
                "name": name,
                "path": raw_path,
                "save_path": self.processed_dir,
                "verbose": self.verbose,
                "force_reload": self.force_reload,
                "label_func": cadence_label_func
            }
            try:
                d = process_score_pitch_spelling(save=False, info=info, feature_type="voice", interval=interval)
                d["note"].cadence = d["note"].y
                d["test"] = gname in self.test_pieces
                torch.save(d, save_fp)
            except:
                if self.verbose:
                    print("Error processing {}".format(raw_path))
            if d is not None:
                data_list.append(d)
        return data_list

