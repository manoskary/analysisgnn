import os
from pathlib import Path

from analysisgnn.data import RNAGraphDataset, RNAplusGraphDataset
from analysisgnn.data.datasets.dlc import DLCGraphDataset, DLCplusGraphDataset
# Removed imports for deleted datasets:
# from analysisgnn.data.datasets.asap import ASAPPitchSpellingGraphDataset
# from analysisgnn.data.datasets.bach_chorales import Bach370ChoralesPitchSpellingGraphDataset
# from analysisgnn.data.datasets.musescore_pop import MusescorePopPitchSpellingGraphDataset
# from analysisgnn.data.datasets.open_string_quartets import OpenStringQuartetsGraphDataset, OpenLiederGraphDataset
# from analysisgnn.data.datasets.kern_datasets import ChopinPreludesGraphDataset, ScarlattiKeybordSonatasGraphDataset
from torch.utils.data import Subset
from analysisgnn.data.data_utils import process_score_pitch_spelling
from analysisgnn.data.remi_bpe_aligner import attach_alignment_to_graph, load_alignment_npz
from analysisgnn.data.data_utils import idx_tuple_to_dict, idx_dict_to_tuple, StandardGraphDataset, CummulativeDataset, struttura_to_inmemory_dataset
from graphmuse.loader import MuseNeighborLoader, transform_to_pyg
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from analysisgnn.data.datasets.cadence import CompleteGraphCadenceDataset, CadenceGraphPGDataset, CadenceSimpleGraphPGDataset
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from analysisgnn.utils.music import CadenceEncoder
import torch
import numpy as np



# Commented out PrEncoderDataModule since it depends on deleted datasets
# class PrEncoderDataModule(LightningDataModule):
#     def __init__(self, num_workers=6, batch_size=16, subgraph_size=100, num_neighbors=[3, 3], device="cpu",
#                  remove_beats=False, remove_measures=False, augment=True, sampling_strategy="musical",
#                  raw_dir=None, force_reload=False, verbose=False, name="AnalysisPretraining"):
#         super(PrEncoderDataModule, self).__init__()
#         datasets = [
#             ASAPPitchSpellingGraphDataset(
#                 raw_dir=raw_dir, force_reload=force_reload, verbose=verbose, num_workers=num_workers),
#             Bach370ChoralesPitchSpellingGraphDataset(
#                 raw_dir=raw_dir, force_reload=force_reload, verbose=verbose, num_workers=num_workers),
#             MusescorePopPitchSpellingGraphDataset(
#                 raw_dir=raw_dir, force_reload=force_reload, verbose=verbose, num_workers=num_workers),
#             OpenStringQuartetsGraphDataset(
#                 raw_dir=raw_dir, force_reload=force_reload, verbose=verbose, num_workers=num_workers),
#             OpenLiederGraphDataset(
#                 raw_dir=raw_dir, force_reload=force_reload, verbose=verbose, num_workers=num_workers),
#             ChopinPreludesGraphDataset(
#                 raw_dir=raw_dir, force_reload=force_reload, verbose=verbose, num_workers=num_workers),
#             ScarlattiKeybordSonatasGraphDataset(
#                 raw_dir=raw_dir, force_reload=force_reload, verbose=verbose, num_workers=num_workers),
#         ]
#         self.test_pieces = sum((d.test_pieces for d in datasets), [])
#         self.dataset = datasets[0] if len(datasets) == 1 else CummulativeDataset(datasets=datasets, name=name, transform=transform_graph)
#         self.augment = augment
#         self.sampling_strategy = sampling_strategy # "neighbor" or "musical"
#         self.batch_size = batch_size
#         self.device = device
#         self.subgraph_size = subgraph_size
#         self.num_workers = num_workers
#         self.num_neighbors = num_neighbors
#         self.remove_beats = remove_beats
#         self.remove_measures = remove_measures
#         self.features = self.dataset[0]["note"].x.shape[-1]
#         assert all(self.features == d[0]["note"].x.shape[-1] for d in datasets)
#         self.num_classes = 35
#         self.metadata = self.dataset[0].metadata()
#
#     def setup(self, stage=None):
#         # if self.augment:
#         #     # Create 3 transpositions per piece
#         #     for dataset in self.datasets:
#         #         dataset.augment(ratio=3)
#
#         trainval_idx = list()
#         self.test_idx = list()
#
#         for i in range(self.dataset.len()):
#             graph = self.dataset[i]
#             if graph["note"].name in self.test_pieces:
#                 self.test_idx.append(i)
#             else:
#                 trainval_idx.append(i)
#
#         if not self.augment:
#             # remove the transposed versions
#             trainval_idx = [i for i in trainval_idx if self.dataset[i]["inverval"] == "P1"]
#
#         if self.remove_beats:
#             m_node = [x for x in self.metadata[0] if x != "beat"]
#             m_edge = [x for x in self.metadata[1] if "beat" not in x]
#             self.metadata = (m_node, m_edge)
#         if self.remove_measures:
#             m_node = [x for x in self.metadata[0] if x != "measure"]
#             m_edge = [x for x in self.metadata[1] if "measure" not in x]
#             self.metadata = (m_node, m_edge)
#         # composers = [g["note"].name.split("-")[0] for g in self.graphs]
#
#         self.train_idx, self.val_idx = train_test_split(trainval_idx, test_size=0.1, random_state=0)
#         print(f"Train: {len(self.train_idx)}, Val: {len(self.val_idx)}, Test: {len(self.test_idx)}")
#
#     def train_dataloader(self):
#         train_graphs = self.dataset[self.train_idx]
#         train_loader = MuseNeighborLoader(train_graphs,
#                                           subgraph_size=self.subgraph_size,
#                                           batch_size=self.batch_size,
#                                           num_neighbors=self.num_neighbors,
#                                           shuffle=False,
#                                           device=self.device,
#                                           num_workers=self.num_workers,
#                                           transform=transform_to_pyg
#                                           )
#         return train_loader
#
#     def val_dataloader(self):
#         val_graphs = self.dataset[self.val_idx]
#         val_loader = MuseNeighborLoader(val_graphs,
#                                         subgraph_size=self.subgraph_size,
#                                         batch_size=self.batch_size,
#                                         num_neighbors=self.num_neighbors,
#                                         shuffle=False,
#                                         subgraph_sample_ratio=1.0,
#                                         device=self.device,
#                                         num_workers=self.num_workers,
#                                         transform=transform_to_pyg
#                                         )
#         return val_loader
#
#     def test_dataloader(self):
#         test_graphs = self.dataset[self.test_idx]
#         test_loader = MuseNeighborLoader(test_graphs,
#                                          subgraph_size=10000,
#                                          batch_size=1,
#                                          num_neighbors=self.num_neighbors,
#                                          shuffle=False,
#                                          subgraph_sample_ratio=1.0,
#                                          device=self.device,
#                                          num_workers=self.num_workers,
#                                          transform=transform_to_pyg
#                                          )
#         return test_loader

def transform_graph(graph):
    voc_edge_index = graph["note", "consecutive", "note"].edge_index
    onset_edge_index = graph["note", "onset", "note"].edge_index
    voice = graph["note"].voice
    staff = graph["note"].staff
    voc_mask = voice[voc_edge_index[0]] == voice[voc_edge_index[1]]
    staff_mask = staff[voc_edge_index[0]] == staff[voc_edge_index[1]]
    onset_staff_mask = staff[onset_edge_index[0]] == staff[onset_edge_index[1]]
    staff_edge_index = torch.cat((voc_edge_index[:, staff_mask], onset_edge_index[:, onset_staff_mask]), dim=1)
    # sort the src nodes of the staff edge index
    staff_edge_index = staff_edge_index[:, staff_edge_index[0].argsort()]
    voc_edge_index = voc_edge_index[:, voc_mask & staff_mask]
    graph["note", "voice", "note"].edge_index = voc_edge_index
    graph["note", "staff", "note"].edge_index = staff_edge_index
    return graph


class AnalysisDataModule(LightningDataModule):
    def __init__(self, num_workers=6, batch_size=16, subgraph_size=100, num_neighbors=[3, 3], device="cpu",
                 remove_beats=False, remove_measures=False, num_cadences=3, augment=True,
                 raw_dir=None, force_reload=False, verbose=False, collection="all", random_split=False,
                 tasks= ["cadence", "rna", "phrase", "ks", "pedal", "staff", "metrical_strength", "is_in_label"],
                 max_samples=None, main_tasks=["cadence", "rna", "all"], feature_type="cadence", training_dataloader_type="combined",                 
                 alignment_dir=None,
                 require_alignment=False,
                 musicbert_embedding_cache_dir=None,
                 require_cached_embeddings=False,
                 ):
        super(AnalysisDataModule, self).__init__()
        # only load the datasets that are needed
        self.training_dataloader_type = training_dataloader_type
        self.datasets = {}
        if feature_type == "cadence":
            for t in main_tasks:
                if t == "cadence":
                    self.datasets[t] = CadenceGraphPGDataset(raw_dir=raw_dir, force_reload=force_reload,
                                                             verbose=verbose, transpose=augment)
                elif t == "rna":
                    self.datasets[t] = RNAplusGraphDataset(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose,
                                                       transpose=augment)
                elif t == "all":
                    self.datasets[t] = DLCplusGraphDataset(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose,
                                                       transpose=augment)
                else:
                    raise ValueError(f"Task {t} is not available")
        else:
            for t in main_tasks:
                if t == "cadence":
                    self.datasets[t] = CadenceSimpleGraphPGDataset(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose, transpose=augment)
                elif t == "rna":
                    self.datasets[t] = RNAGraphDataset(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose, transpose=augment)
                elif t == "all":
                    self.datasets[t] = DLCGraphDataset(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose, transpose=augment)
                else:
                    raise ValueError(f"Task {t} is not available")

        # Join all the lists dataset.test_pieces
        self.random_split = random_split
        self.cadence_encoder = CadenceEncoder()
        self.roman_numeral_encoder = None
        self.phrase_encoder = None
        self.ps_encoder = None
        self.ks_encoder = None
        self.num_cadences = num_cadences
        self.batch_size = batch_size
        self.device = device
        self.augment = augment
        self.verbose = verbose
        self.tasks = tasks
        self.main_tasks = main_tasks
        self.subgraph_size = subgraph_size
        self.num_workers = num_workers
        self.num_neighbors = num_neighbors
        self.remove_beats = remove_beats
        self.remove_measures = remove_measures
        self.alignment_dir = alignment_dir
        self.require_alignment = require_alignment
        self.musicbert_embedding_cache_dir = musicbert_embedding_cache_dir
        self.require_cached_embeddings = require_cached_embeddings

        if max_samples is not None:
            for k in list(self.datasets.keys()):
                indices = list(range(len(self.datasets[k])))
                if not self.augment:
                    keep = []
                    for i in indices:
                        graph = self.datasets[k][i]
                        transposition = getattr(graph, "transposition", None)
                        if transposition is None:
                            try:
                                transposition = graph["transposition"]
                            except Exception:
                                transposition = None
                        if transposition is None:
                            try:
                                transposition = graph["interval"]
                            except Exception:
                                transposition = None
                        if transposition is None or transposition == "P1":
                            keep.append(i)
                    indices = keep
                if not indices:
                    if self.verbose:
                        print(f"Dataset {k} has no samples after transposition filter.")
                    self.datasets[k] = Subset(self.datasets[k], [])
                    continue
                if len(indices) > max_samples:
                    perm = torch.randperm(len(indices))[:max_samples].tolist()
                    indices = [indices[i] for i in perm]
                self.datasets[k] = Subset(self.datasets[k], indices)
        if self.alignment_dir is not None or self.musicbert_embedding_cache_dir is not None:
            self._setup_alignment_transforms()
        # assert that the features are the same
        key = list(self.datasets.keys())[0]
        self.features = self.datasets[key][0]["note"].x.shape[-1]
        self.metadata = self._process_graph_metadata(self.datasets[key][0].metadata())
        self.current_val_tasks = [] if training_dataloader_type != "combined" else self.main_tasks
        self.current_task = None if training_dataloader_type != "combined" else self.main_tasks
        self._loader_log_calls = {"train": 0, "val": 0, "test": 0}

    def _process_graph_metadata(self, metadata):
        nodes, edges = metadata
        if self.remove_beats:
            nodes = [n for n in nodes if n != "beat"]
            edges = [e for e in edges if "beat" not in e]
        if self.remove_measures:
            nodes = [n for n in nodes if n != "measure"]
            edges = [e for e in edges if "measure" not in e]
        return (nodes, edges)

    def set_task(self, text):
        if self.training_dataloader_type != "combined":
            assert text in self.main_tasks, f"Task {text} not available"
            self.current_task = text
            self.current_val_tasks.append(text)

    @property
    def num_classes(self):
        if self.num_cadences < self.cadence_encoder.encode_dim:
            return self.num_cadences
        return self.cadence_encoder.encode_dim

    def prepare_data(self):
        pass

    def _alignment_name_for_graph(self, graph):
        name = getattr(graph, "name", None)
        if name is None:
            try:
                name = graph["name"]
            except Exception:
                name = None
        if name is None:
            return None

        interval = getattr(graph, "transposition", None)
        if interval is None:
            interval = getattr(graph, "interval", None)
        if interval is None:
            try:
                interval = graph["transposition"]
            except Exception:
                interval = None
        if interval is None:
            try:
                interval = graph["interval"]
            except Exception:
                interval = None

        if interval and interval != "P1":
            return f"{name}_{interval}"
        return str(name)

    def _wrap_dataset_transform(self, dataset, fn):
        base = dataset.dataset if isinstance(dataset, Subset) else dataset
        previous = getattr(base, "transform", None)
        if previous is None:
            base.transform = fn
            return

        def composed(graph, previous=previous):
            graph = previous(graph)
            return fn(graph)

        base.transform = composed

    def _setup_alignment_transforms(self):
        alignment_dir = Path(self.alignment_dir) if self.alignment_dir is not None else None
        if alignment_dir is not None and not alignment_dir.exists():
            raise ValueError(f"Alignment directory not found: {self.alignment_dir}")
        embedding_cache_dir = (
            Path(self.musicbert_embedding_cache_dir)
            if self.musicbert_embedding_cache_dir is not None
            else None
        )
        if embedding_cache_dir is not None and not embedding_cache_dir.exists():
            raise ValueError(f"MusicBERT embedding cache directory not found: {self.musicbert_embedding_cache_dir}")

        def add_note_idx(graph):
            if "note" in getattr(graph, "node_types", []):
                note_store = graph["note"]
                if not hasattr(note_store, "note_idx"):
                    note_store.note_idx = torch.arange(note_store.num_nodes)
            return graph

        def attach_alignment(graph):
            if alignment_dir is None:
                return graph
            if hasattr(graph, "input_ids") and hasattr(graph, "token2note"):
                return graph
            align_name = self._alignment_name_for_graph(graph)
            if not align_name:
                return graph
            alignment_path = alignment_dir / f"{align_name}.npz"
            if not alignment_path.exists():
                return graph
            alignment = load_alignment_npz(str(alignment_path))
            attach_alignment_to_graph(graph, alignment)
            return graph

        def attach_cached_note_embeddings(graph):
            if embedding_cache_dir is None:
                return graph
            if "note" not in getattr(graph, "node_types", []):
                return graph
            note_store = graph["note"]
            if hasattr(note_store, "musicbert_note_embeddings"):
                return graph

            align_name = self._alignment_name_for_graph(graph)
            if not align_name:
                return graph

            cache_path = embedding_cache_dir / f"{align_name}.npz"
            if not cache_path.exists():
                return graph

            with np.load(cache_path) as payload:
                if "note_embeddings" not in payload:
                    return graph
                note_embeddings = payload["note_embeddings"]
            if note_embeddings.ndim != 2:
                raise ValueError(
                    f"Invalid cached MusicBERT embeddings shape {note_embeddings.shape} at {cache_path}"
                )
            note_store.musicbert_note_embeddings = torch.from_numpy(note_embeddings)
            return graph

        for dataset in self.datasets.values():
            self._wrap_dataset_transform(dataset, add_note_idx)
            self._wrap_dataset_transform(dataset, attach_alignment)
            self._wrap_dataset_transform(dataset, attach_cached_note_embeddings)

    def _filter_datasets_by_alignment(self):
        if self.alignment_dir is None:
            return
        alignment_dir = Path(self.alignment_dir)
        if not alignment_dir.exists():
            raise ValueError(f"Alignment directory not found: {self.alignment_dir}")
        alignment_names = {p.stem for p in alignment_dir.glob("*.npz")}

        removed = []
        for key in list(self.datasets.keys()):
            dataset = self.datasets[key]
            aligned_idx = []
            missing = 0
            for i in range(len(dataset)):
                graph = dataset[i]
                align_name = self._alignment_name_for_graph(graph)
                if align_name and align_name in alignment_names:
                    aligned_idx.append(i)
                else:
                    missing += 1
            if not aligned_idx:
                removed.append(key)
                del self.datasets[key]
                continue
            if missing and self.verbose:
                print(f"Dataset {key}: missing {missing} alignment(s); keeping {len(aligned_idx)} graphs.")
            if missing:
                self.datasets[key] = Subset(dataset, aligned_idx)

        if removed:
            self.main_tasks = [task for task in self.main_tasks if task in self.datasets]
            if self.training_dataloader_type == "combined":
                self.current_task = self.main_tasks
                self.current_val_tasks = self.main_tasks
            if self.verbose:
                print(f"Removed datasets without alignments: {removed}")
        if not self.datasets:
            raise ValueError(f"No datasets remain after filtering by alignments in {alignment_dir}")

    def _filter_datasets_by_embedding_cache(self):
        if self.musicbert_embedding_cache_dir is None:
            return
        cache_dir = Path(self.musicbert_embedding_cache_dir)
        if not cache_dir.exists():
            raise ValueError(f"MusicBERT embedding cache directory not found: {cache_dir}")
        cache_names = {p.stem for p in cache_dir.glob("*.npz")}

        removed = []
        for key in list(self.datasets.keys()):
            dataset = self.datasets[key]
            cached_idx = []
            missing = 0
            for i in range(len(dataset)):
                graph = dataset[i]
                cache_name = self._alignment_name_for_graph(graph)
                if cache_name and cache_name in cache_names:
                    cached_idx.append(i)
                else:
                    missing += 1
            if not cached_idx:
                removed.append(key)
                del self.datasets[key]
                continue
            if missing and self.verbose:
                print(
                    f"Dataset {key}: missing {missing} cached MusicBERT embedding(s); "
                    f"keeping {len(cached_idx)} graphs."
                )
            if missing:
                self.datasets[key] = Subset(dataset, cached_idx)

        if removed:
            self.main_tasks = [task for task in self.main_tasks if task in self.datasets]
            if self.training_dataloader_type == "combined":
                self.current_task = self.main_tasks
                self.current_val_tasks = self.main_tasks
            if self.verbose:
                print(f"Removed datasets without cached MusicBERT embeddings: {removed}")
        if not self.datasets:
            raise ValueError(
                f"No datasets remain after filtering by cached MusicBERT embeddings in {cache_dir}"
            )

    def _filter_datasets_by_transposition(self, interval: str = "P1"):
        removed = []
        for key in list(self.datasets.keys()):
            dataset = self.datasets[key]
            keep_idx = []
            for i in range(len(dataset)):
                graph = dataset[i]
                transposition = getattr(graph, "transposition", None)
                if transposition is None:
                    try:
                        transposition = graph["transposition"]
                    except Exception:
                        transposition = None
                if transposition is None:
                    try:
                        transposition = graph["interval"]
                    except Exception:
                        transposition = None
                if transposition is None or transposition == interval:
                    keep_idx.append(i)

            if not keep_idx:
                removed.append(key)
                del self.datasets[key]
                continue
            if len(keep_idx) != len(dataset):
                self.datasets[key] = Subset(dataset, keep_idx)

        if removed:
            self.main_tasks = [task for task in self.main_tasks if task in self.datasets]
            if self.training_dataloader_type == "combined":
                self.current_task = self.main_tasks
                self.current_val_tasks = self.main_tasks
            if self.verbose:
                print(f"Removed datasets without '{interval}' transposition: {removed}")
        if not self.datasets:
            raise ValueError(f"No datasets remain after filtering transpositions ({interval}).")

    @staticmethod
    def _graph_interval(graph):
        interval = getattr(graph, "transposition", None)
        if interval is None:
            interval = getattr(graph, "interval", None)
        if interval is None:
            try:
                interval = graph["transposition"]
            except Exception:
                interval = None
        if interval is None:
            try:
                interval = graph["interval"]
            except Exception:
                interval = None
        return interval if interval is not None else "P1"

    def _interval_counts(self, dataset, indices):
        counts = {}
        for idx in indices:
            graph = dataset[idx]
            interval = self._graph_interval(graph)
            counts[interval] = counts.get(interval, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[0]))

    def _log_split_interval_stats(self):
        for key, dataset in self.datasets.items():
            train_counts = self._interval_counts(dataset, self.train_idx[key])
            val_counts = self._interval_counts(dataset, self.val_idx[key])
            test_counts = self._interval_counts(dataset, self.test_idx[key])
            print(
                f"[split-stats] dataset={key} "
                f"train={len(self.train_idx[key])} val={len(self.val_idx[key])} test={len(self.test_idx[key])} "
                f"train_intervals={train_counts} val_intervals={val_counts} test_intervals={test_counts}"
            )

    def _log_loader_sizes(self, stage: str, loaders):
        self._loader_log_calls[stage] += 1
        epoch = getattr(getattr(self, "trainer", None), "current_epoch", None)
        prefix = f"[loader-stats] stage={stage} call={self._loader_log_calls[stage]}"
        if epoch is not None:
            prefix += f" epoch={epoch}"

        if isinstance(loaders, dict):
            details = []
            for task_name, loader in loaders.items():
                try:
                    num_batches = len(loader)
                except Exception:
                    num_batches = "?"
                details.append(f"{task_name}:graphs={len(loader.dataset)},batches={num_batches}")
            print(f"{prefix} {'; '.join(details)}")
            return

        try:
            num_batches = len(loaders)
        except Exception:
            num_batches = "?"
        dataset_size = len(loaders.dataset) if hasattr(loaders, "dataset") else "?"
        print(f"{prefix} graphs={dataset_size} batches={num_batches}")

    def setup(self, stage=None):
        if self.require_alignment:
            self._filter_datasets_by_alignment()
        if self.require_cached_embeddings:
            self._filter_datasets_by_embedding_cache()
        if not self.augment:
            self._filter_datasets_by_transposition("P1")
        # random split
        self.train_idx = {}
        self.val_idx = {}
        self.test_idx = {}
        for k in self.datasets.keys():
            # Test set are files with property test=True
            if self.random_split:
                trainval_idx, self.test_idx[k] = train_test_split(
                    range(len(self.datasets[k])), test_size=0.2, random_state=0
                )
            else:
                test_mask = np.array([g["test"] for g in self.datasets[k]])
                if test_mask.sum() > 0:
                    trainval_idx = np.where(~test_mask)[0]
                    self.test_idx[k] = np.where(test_mask)[0]
                else:
                    if self.verbose:
                        print(f"No test files found in dataset {k}; falling back to random split.")
                    trainval_idx, self.test_idx[k] = train_test_split(
                        range(len(self.datasets[k])), test_size=0.2, random_state=0
                    )

            self.train_idx[k], self.val_idx[k] = train_test_split(trainval_idx, test_size=0.1, random_state=0)

        if self.verbose:
            for k in self.datasets.keys():
                print(f"Datataset {k} | Train: {len(self.train_idx[k])}, Val: {len(self.val_idx[k])}, Test: {len(self.test_idx[k])}")
        self._log_split_interval_stats()

    def train_dataloader(self):
        transform = self._build_transform()
        if self.training_dataloader_type == "sequential":
            train_graphs = Subset(self.datasets[self.trainer.model.current_task], self.train_idx[self.trainer.model.current_task])
            train_loader = MuseNeighborLoader(train_graphs,
                                              subgraph_size=self.subgraph_size,
                                              batch_size=self.batch_size,
                                              num_neighbors=self.num_neighbors,
                                              device=self.device,
                                              num_workers=self.num_workers,
                                              subgraph_sample_ratio=0.5,
                                              transform=transform
                                              )
            self._log_loader_sizes("train", train_loader)
            return train_loader
        elif self.training_dataloader_type == "combined":
            train_loaders = {}
            for mt in self.main_tasks:
                train_graphs = Subset(self.datasets[mt], self.train_idx[mt])
                train_loaders[mt] = MuseNeighborLoader(train_graphs,
                                                subgraph_size=self.subgraph_size,
                                                batch_size=self.batch_size // len(self.main_tasks),
                                                num_neighbors=self.num_neighbors,
                                                device=self.device,
                                                num_workers=self.num_workers,
                                                subgraph_sample_ratio=0.5,
                                                transform=transform
                                                )
            self._log_loader_sizes("train", train_loaders)
            return CombinedLoader(train_loaders, "min_size")

    def val_dataloader(self):
        transform = self._build_transform()
        val_loaders = {}
        for mt in self.main_tasks:
            val_graphs = Subset(self.datasets[mt], self.val_idx[mt])
            val_loaders[mt] = MuseNeighborLoader(val_graphs,
                                            subgraph_size=self.subgraph_size,
                                            batch_size=self.batch_size // len(self.main_tasks),
                                            num_neighbors=self.num_neighbors,
                                            device=self.device,
                                            num_workers=self.num_workers,
                                            subgraph_sample_ratio=0.5,
                                            transform=transform
                                            )
        self._log_loader_sizes("val", val_loaders)
        return CombinedLoader(val_loaders, "max_size")

    def test_dataloader(self):
        transform = self._build_transform()
        test_loaders = {}
        for mt in self.main_tasks:
            test_graphs = Subset(self.datasets[mt], self.test_idx[mt])
            test_loaders[mt] = MuseNeighborLoader(test_graphs,
                                             subgraph_size=10000,
                                             batch_size=1,
                                             num_neighbors=self.num_neighbors,
                                             device=self.device,
                                             num_workers=0,
                                             subgraph_sample_ratio=0.5,
                                             transform=transform,
                                             shuffle=False
                                             )
        self._log_loader_sizes("test", test_loaders)
        return CombinedLoader(test_loaders, "max_size")

    def _build_transform(self):
        def transform(graph, num_hops=None, *args, **kwargs):
            if num_hops is None:
                try:
                    num_hops = len(self.num_neighbors)
                except Exception:
                    num_hops = 1
            graph = transform_to_pyg(graph, num_hops)
            return graph

        return transform
