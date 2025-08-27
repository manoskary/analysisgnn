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

        if max_samples is not None:
            for k in self.datasets.keys():
                # shuffle the indices and keep only the first max_samples
                self.datasets[k] = Subset(self.datasets[k], torch.randperm(len(self.datasets[k]))[:max_samples])
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
        # assert that the features are the same
        key = list(self.datasets.keys())[0]
        self.features = self.datasets[key][0]["note"].x.shape[-1]
        self.metadata = self._process_graph_metadata(self.datasets[key][0].metadata())
        self.current_val_tasks = [] if training_dataloader_type != "combined" else self.main_tasks
        self.current_task = None if training_dataloader_type != "combined" else self.main_tasks

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

    def setup(self, stage=None):
        # random split
        self.train_idx = {}
        self.val_idx = {}
        self.test_idx = {}
        for k in self.datasets.keys():
            # Test set are files with property test=True
            if self.random_split:
                trainval_idx, self.test_idx[k] = train_test_split(range(len(self.datasets[k])), test_size=0.2, random_state=0)
            else:
                test_mask = np.array([g["test"] for g in self.datasets[k]])
                assert test_mask.sum() > 0, f"No test files found in dataset {k}"
                trainval_idx = np.where(~test_mask)[0]
                self.test_idx[k] = np.where(test_mask)[0]

            if not self.augment:
                # remove the transposed versions
                trainval_idx = [i for i in trainval_idx if self.datasets[k][i]["transposition"] == "P1"]

            self.train_idx[k], self.val_idx[k] = train_test_split(trainval_idx, test_size=0.1, random_state=0)

        if self.verbose:
            for k in self.datasets.keys():
                print(f"Datataset {k} | Train: {len(self.train_idx[k])}, Val: {len(self.val_idx[k])}, Test: {len(self.test_idx[k])}")

    def train_dataloader(self):
        if self.training_dataloader_type == "sequential":
            train_graphs = Subset(self.datasets[self.trainer.model.current_task], self.train_idx[self.trainer.model.current_task])
            train_loader = MuseNeighborLoader(train_graphs,
                                              subgraph_size=self.subgraph_size,
                                              batch_size=self.batch_size,
                                              num_neighbors=self.num_neighbors,
                                              device=self.device,
                                              num_workers=self.num_workers,
                                              subgraph_sample_ratio=0.5,
                                              transform=transform_to_pyg
                                              )
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
                                                transform=transform_to_pyg
                                                )
            return CombinedLoader(train_loaders, "min_size")

    def val_dataloader(self):
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
                                            transform=transform_to_pyg
                                            )
        return CombinedLoader(val_loaders, "max_size")

    def test_dataloader(self):
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
                                             transform=transform_to_pyg,
                                             shuffle=False
                                             )
        return CombinedLoader(test_loaders, "max_size")

