from analysisgnn.utils.graph import *
import os
from analysisgnn.data.dataset import BuiltinDataset, StrutturaDataset, get_download_dir
from analysisgnn.descriptors import pitch_spelling_features, select_features
import partitura as pt
from analysisgnn.utils import hetero_graph_from_note_array, HeteroScoreGraph, load_score_hgraph
from tqdm import tqdm
from joblib import Parallel, delayed
# Removed performance import: from .performance import MatchGraphDataset
from numpy.lib.recfunctions import structured_to_unstructured
from tqdm.contrib.concurrent import process_map
from graphmuse import create_score_graph
from analysisgnn.models.pitch_spelling import PitchEncoder, KeySignatureEncoder
from analysisgnn.utils import transpose_note_array
from analysisgnn.data.data_utils import process_score_pitch_spelling
from torch_geometric.data import InMemoryDataset, HeteroData
from typing import List, Union, Tuple
import shutil



class ASAPDataset(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True):
        url = "https://github.com/CPJKU/asap-dataset.git"
        self.scores = list()
        self.composers = list()
        self.score_names = list()
        self.performances = list()
        self.performer_names = list()
        self.match = list()
        self.joint_ps = list()
        self.problematic_pieces = ["Chopin-Etudes_op_25-2", 'Ravel-Gaspard_de_la_Nuit-1_Ondine',
                                   "Schumann-Kreisleriana-3", "Schubert-Impromptu_op142-3",
                                   "Beethoven-Piano_Sonatas-7-1", "Scriabin-Sonatas-5",
                                   "Debussy-Pour_le_Piano-1"]
        super(ASAPDataset, self).__init__(
            name="ASAPDataset",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            clone=True,
            branch="main")
        self.process()

    def process(self):
        self.scores = list()
        self.composers = list()
        self.score_names = list()
        self.performances = list()
        self.performer_names = list()
        self.match = list()
        self.joint_ps = list()
        for root, dirs, files in os.walk(self.save_path):
            for fn in files:
                if fn.endswith("xml_score.musicxml"):
                    self.scores.append(os.path.join(root, fn))
                    # The direct subfolder of the raw_dir is the name of the composer
                    self.composers.append(os.path.relpath(root, self.raw_path).split(os.sep)[0])
                    self.score_names.append("-".join(os.path.relpath(root, self.raw_path).split(os.sep)))
                if fn.endswith(".mid") and fn != "midi_score.mid":
                    self.performances.append(os.path.join(root, fn))
                    self.performer_names.append(os.path.splitext(os.path.basename(fn))[0])
                if fn.endswith(".match"):
                    name = "-".join(root.replace(self.raw_path, "").split('/'))[1:]
                    if name not in self.problematic_pieces and os.path.join(root, fn) not in self.match:
                        self.match.append(os.path.join(root, fn))
        # For each match get the corresponding score
        matched_scores = [os.path.join(os.path.dirname(match_file), "xml_score.musicxml") for match_file in self.match]
        # Zip them together
        self.match = list(zip(self.match, matched_scores))


    def has_cache(self):
        if os.path.exists(self.save_path):
            return True

        return False


class ASAPGraphDataset(StrutturaDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, n_jobs=12, include_measures=False, max_size=200):
        self.asap_dataset = ASAPDataset(raw_dir=raw_dir)
        self.n_jobs = n_jobs
        self.graphs = []
        self.max_size = max_size
        self.stage = "validate"
        self._force_reload = force_reload
        self.include_measures = include_measures
        self.prob_scores = ["Chopin-Etudes_op_25-2", 'Ravel-Gaspard_de_la_Nuit-1_Ondine']
        self.composer_to_label = dict()
        self.label_to_composer = dict()
        if verbose:
            print("Loaded Asap Dataset Successfully, now processing...")
        super(ASAPGraphDataset, self).__init__(
            name="ASAPGraphDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def process(self):
        zfiles = zip(self.asap_dataset.scores, self.asap_dataset.composers, self.asap_dataset.score_names)
        process_map(self._process_score, zfiles, max_workers=self.n_jobs)
        self.load()

    def _process_score(self, data):
        score_fn, composer, score_name = data
        if score_name in self.prob_scores:
            return
        if self._force_reload or not (os.path.exists(os.path.join(self.save_path, score_name))):
            score = pt.load_score(score_fn)
            note_array = score.note_array(include_time_signature=True)
            note_features = select_features(note_array, "voice")
            nodes, edges = hetero_graph_from_note_array(note_array)
            hg = HeteroScoreGraph(
                note_features,
                edges,
                name=score_name,
                labels=None,
                note_array=note_array,
            )
            hg.y = composer
            measures = score[np.array([p._quarter_durations[0] for p in score]).argmax()].measures
            hg.add_beat_nodes()
            hg.add_measure_nodes(measures)

            hg.save(self.save_path)
        return

    def save(self):
        pass

    def load(self):
        # Filter for composers
        composers, counts = np.unique(np.array(self.asap_dataset.composers), return_counts=True)
        rejected_composers = composers[np.where(counts < 4)]
        for fn in os.listdir(self.save_path):
            if fn in self.prob_scores:
                continue
            path_graph = os.path.join(self.save_path, fn)
            graph = load_score_hgraph(path_graph, fn)
            composer = graph.y
            # filter composer list
            if composer in rejected_composers:
                continue
            if composer not in self.composer_to_label.keys():
                max_i = max(self.composer_to_label.values()) if len(self.composer_to_label) > 0 else -1
                self.composer_to_label[composer] = max_i + 1
                self.label_to_composer[max_i + 1] = composer
            self.graphs.append(graph)

    def has_cache(self):
        if all(
                [os.path.exists(os.path.join(self.save_path, path))
                 for path in self.asap_dataset.score_names]
        ):
            return True
        return False

    def __len__(self):
        return len(self.graphs)


    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.get_graph_attr(idx, self.stage == "train")
        return [self.get_graph_attr(i, self.stage == "train") for i in idx]

    def set_split(self, stage="train"):
        self.stage = stage

    def get_graph_attr(self, idx, batch=True):
        out = dict()
        if self.graphs[idx].x.size(0) > self.max_size and batch:
            random_idx = random.randint(0, self.graphs[idx].x.size(0) - self.max_size)
            indices = torch.arange(random_idx, random_idx + self.max_size)
            edge_indices = torch.isin(self.graphs[idx].edge_index[0], indices) & torch.isin(
                self.graphs[idx].edge_index[1], indices)
            out["x"] = self.graphs[idx].x[indices]
            out["edge_index"] = self.graphs[idx].edge_index[:, edge_indices] - random_idx
            out["y"] = self.composer_to_label[self.graphs[idx].y]
            out["edge_type"] = self.graphs[idx].edge_type[edge_indices]
            out["note_array"] = structured_to_unstructured(
                self.graphs[idx].note_array[
                    ["pitch", "onset_div", "duration_div", "onset_beat", "duration_beat", "ts_beats"]]
            )[indices]
            out["name"] = self.graphs[idx].name
            if self.include_measures:
                measure_edges = torch.tensor(self.graphs[idx].measure_edges)
                measure_nodes = torch.tensor(self.graphs[idx].measure_nodes).squeeze()
                beat_edges = torch.tensor(self.graphs[idx].beat_edges)
                beat_nodes = torch.tensor(self.graphs[idx].beat_nodes).squeeze()
                beat_edge_indices = torch.isin(beat_edges[0], indices)
                beat_node_indices = torch.isin(beat_nodes, torch.unique(beat_edges[1][beat_edge_indices]))
                min_beat_idx = torch.where(beat_node_indices)[0].min()
                max_beat_idx = torch.where(beat_node_indices)[0].max()
                measure_edge_indices = torch.isin(measure_edges[0], indices)
                measure_node_indices = torch.isin(measure_nodes, torch.unique(measure_edges[1][measure_edge_indices]))
                min_measure_idx = torch.where(measure_node_indices)[0].min()
                max_measure_idx = torch.where(measure_node_indices)[0].max()
                out["beat_nodes"] = beat_nodes[min_beat_idx:max_beat_idx + 1] - min_beat_idx
                out["beat_edges"] = torch.vstack(
                    (beat_edges[0, beat_edge_indices] - random_idx, beat_edges[1, beat_edge_indices] - min_beat_idx))
                out["measure_nodes"] = measure_nodes[min_measure_idx:max_measure_idx + 1] - min_measure_idx
                out["measure_edges"] = torch.vstack((measure_edges[0, measure_edge_indices] - random_idx,
                                                     measure_edges[1, measure_edge_indices] - min_measure_idx))
        else:
            out["x"] = self.graphs[idx].x
            out["edge_index"] = self.graphs[idx].edge_index
            out["y"] = self.composer_to_label[self.graphs[idx].y]
            out["edge_type"] = self.graphs[idx].edge_type
            out["note_array"] = structured_to_unstructured(
                self.graphs[idx].note_array[
                    ["pitch", "onset_div", "duration_div", "onset_beat", "duration_beat", "ts_beats"]]
            )
            out["name"] = self.graphs[idx].name
            if self.include_measures:
                out["beat_nodes"] = torch.tensor(self.graphs[idx].beat_nodes)
                out["beat_edges"] = torch.tensor(self.graphs[idx].beat_edges)
                out["measure_nodes"] = torch.tensor(self.graphs[idx].measure_nodes)
                out["measure_edges"] = torch.tensor(self.graphs[idx].measure_edges)
        return out

    @property
    def features(self):
        return self.graphs[0].x.shape[1]

    @property
    def n_classes(self):
        return len(self.composer_to_label)


class ASAPPitchSpellingDataset(StrutturaDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True):
        self.asap_dataset = ASAPDataset(raw_dir=raw_dir)
        if verbose:
            print("Loaded Asap Dataset Successfully, now processing...")
        if os.path.exists(os.path.join(self.raw_dir, "ASAPGraphPitchSpellingDataset")):
            self.X, self.y = self.get_cached_files(os.path.join(self.raw_dir, "ASAPGraphPitchSpellingDataset"))
        super(ASAPPitchSpellingDataset, self).__init__(
            name="ASAPPitchSpellingDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def process(self):
        pass

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True
        return False

    def get_cached_files(self, path):
        import shutil
        for fn in os.listdir(path):
            shutil.copy(os.path.join(path, fn, "x.npy"), os.path.join(self.save_dir, fn, "x.npy"))
            shutil.copy(os.path.join(path, fn, "y.npy"), os.path.join(self.save_dir, fn, "y.npy"))

    def save(self):
        """save the graph list and the labels"""
        def asap_rec_save(par_dir, doc_type, save_dir, result=[]):
            for cdir in os.listdir(par_dir):
                path = os.path.join(par_dir, cdir)
                if os.path.isdir(path):
                    result = asap_rec_save(path, doc_type, save_dir, result)
                else:
                    if path.endswith(doc_type):
                        try:
                            if os.path.dirname(os.path.dirname(par_dir)) != "ASAPDataset":
                                name = os.path.basename(
                                    os.path.dirname(os.path.dirname(par_dir))) + "_" + os.path.basename(
                                    os.path.dirname(par_dir)) + "_" + os.path.basename(par_dir)
                            else:
                                name = os.path.basename(
                                    os.path.basename(os.path.dirname(par_dir)) + "_" + os.path.basename(par_dir))
                            if not os.path.exists(os.path.join(save_dir, name)):
                                os.makedirs(os.path.join(save_dir, name))
                                part = pt.score.merge_parts(pt.load_score(path))
                                part = pt.score.unfold_part_maximal(part)
                                features, _ = pitch_spelling_features(part)
                                labels = part.note_array(include_pitch_spelling=True, include_key_signature=True)[["step", "alter", "ks_fifths", "ks_mode"]]
                                with open(os.path.join(save_dir, name, "x.npy"), "wb") as f:
                                    np.save(f, features)
                                with open(os.path.join(save_dir, name, "y.npy"), "wb") as f:
                                    np.save(f, labels)
                                result.append(name)
                        except:
                            print("Piece {} failed".format(name))
                            pass
            return result
        self.scores = asap_rec_save(os.path.join(self.raw_dir, "ASAPDataset"), ".musicxml", os.path.join(self.save_dir, self.name))

    def load(self):
        self.graphs = list()
        self.scores = [fn for fn in os.listdir(self.save_dir)]

    def __getitem__(self, idx):
        feature_path = os.path.join(self.save_path, self.scores[idx], "x.npy")
        label_path = os.path.join(self.save_path, self.features[idx], "y.npy")
        feature = self.read_file(feature_path)
        label = self.read_file(label_path)
        return feature, label, len(feature)

    def read_file(self, path):
        return torch.from_numpy(np.load(path))

    def __len__(self):
        return len(self.scores)

    @property
    def save_name(self):
        return self.name


ALT_TO_ACC = {
    0: "",
    1: "#",
    2: "##",
    -1: "-",
    -2: "--"
}


class ASAPPitchSpellingGraphDataset(InMemoryDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, num_workers=4):
        self.base_dataset = ASAPDataset(raw_dir=raw_dir)
        if verbose:
            print("Loaded Asap Dataset Successfully, now processing...")
        root = os.path.join(get_download_dir(),
                            "ASAPPitchSpellingGraphDataset") if raw_dir is None else os.path.join(
            raw_dir, "ASAPPitchSpellingGraphDataset")
        self.verbose = verbose
        self.force_reload = force_reload
        self.num_workers = num_workers
        transform, pre_filter, pre_transform = None, None, None
        self.num_workers = num_workers
        self.prob_pieces = ["Ravel-Gaspard_de_la_Nuit-1_Ondine", "Schubert-Piano_Sonatas-664-2", "Chopin-Etudes_op_25-2"]
        self.test_pieces = [
            'Bach-Prelude-bwv_887', 'Ravel-Pavane', 'Rachmaninoff-Preludes_op_32-10', 'Bach-Fugue-bwv_864',
            'Bach-Fugue-bwv_863', 'Schubert-Impromptu_op142-1', 'Schumann-Kreisleriana-2',
            'Bach-Fugue-bwv_864', 'Chopin-Polonaises-53', 'Bach-Prelude-bwv_880',
            'Liszt-2_La_campanella', 'Bach-Fugue-bwv_892',
            'Schumann-Kreisleriana-5', 'Chopin-Berceuse_op_57', 'Schubert-Piano_Sonatas-894-2',
            'Chopin-Etudes_op_25-5', 'Schubert-Impromptu_op142-3', 'Bach-Prelude-bwv_867', 'Bach-Fugue-bwv_891',
            'Bach-Fugue-bwv_846', 'Prokofiev-Toccata', 'Schumann-Toccata', 'Haydn-Keyboard_Sonatas-39-2',
            'Beethoven-Piano_Sonatas-21-2', 'Schumann-Kreisleriana-3', 'Ravel-Miroirs-3_Une_Barque',
            'Bach-Fugue-bwv_885', 'Bach-Fugue-bwv_889', 'Beethoven-Piano_Sonatas-18-3', 'Bach-Prelude-bwv_870',
            'Chopin-Sonata_3-2nd', 'Beethoven-Piano_Sonatas-8-2', 'Chopin-Etudes_op_25-12',
            'Beethoven-Piano_Sonatas-7-1', 'Beethoven-Piano_Sonatas-21-3']
        super(ASAPPitchSpellingGraphDataset, self).__init__(root, transform, pre_filter)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        fns = []
        if not os.path.exists(self.raw_dir) and not os.path.isdir(self.raw_dir):
            os.makedirs(self.raw_dir)
        for ofp, ofn in zip(self.base_dataset.scores, self.base_dataset.score_names):
            fn = ofn + ".mscx"
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

    def process(self):
        data_list = []
        for raw_path in self.raw_paths:
            save_fp = os.path.join(self.processed_dir, os.path.splitext(os.path.basename(raw_path))[0] + ".pt")
            if os.path.exists(save_fp):
                d = torch.load(save_fp)
                data_list.append(d)
                continue
            if os.path.splitext(os.path.basename(raw_path))[0] in self.prob_pieces:
                continue
            try:
                d = process_score_pitch_spelling({"name": os.path.splitext(os.path.basename(raw_path))[0],
                                                  "path": raw_path, "save_path": self.processed_dir})
            except:
                print("Error processing {}".format(raw_path))
                continue
            if d is not None:
                data_list.append(d)

        self.save(data_list, self.processed_paths[0])


class ASAPGraphPerformanceDataset(StrutturaDataset):
    r"""The citation graph dataset, including cora, citeseer and pubmeb.
    Nodes mean authors and edges mean citation relationships.
    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if Asap Dataset contining the scores is already available otherwise it will download it.
        Default: ~/.struttura/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """

    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, nprocs=4):
        self.asap_dataset = ASAPDataset(raw_dir=raw_dir)
        self.nprocs = nprocs
        if verbose:
            print("Loaded Asap Dataset Successfully, now processing...")
        self.graphs = list()
        super(ASAPGraphPerformanceDataset, self).__init__(
            name="ASAPGraphPerformanceDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def process(self):
        def asap_rec_save(par_dir, doc_type, save_dir, result=[]):
            for cdir in os.listdir(par_dir):
                path = os.path.join(par_dir, cdir)
                if os.path.isdir(path):
                    result = asap_rec_save(path, doc_type, save_dir, result)
                else:
                    if path.endswith(doc_type):
                        pdir = os.path.dirname(par_dir)
                        try:
                            if os.path.dirname(os.path.dirname(pdir)) != "ASAPDataset":
                                name = os.path.basename(
                                    os.path.dirname(os.path.dirname(pdir))) + "_" + os.path.basename(
                                    os.path.dirname(pdir)) + "_" + os.path.basename(pdir)
                            else:
                                name = os.path.basename(
                                    os.path.basename(os.path.dirname(pdir)) + "_" + os.path.basename(pdir))
                            save_name = os.path.basename(par_dir).split("_note_alignments")[0]
                            if not os.path.exists(os.path.join(path, name, save_name)):
                                sfn = os.path.join(pdir, "xml_score.musicxml")
                                pfn = os.path.join(pdir, save_name + ".mid")
                                afn = path
                                result.append((sfn, pfn, afn, save_name, name))
                        except:
                            print("Graph Creation failed on {}".format(path))
            return result

        def gcreate(sfn, pfn, afn, save_name, gname):
            try:
                g = pgraph_from_part(sfn, pfn, afn, save_name)
                g.save(os.path.join(self.save_dir, self.name, gname))
                return g
            except:
                print("Graph Creation failed on {}".format(sfn))

        load_data = asap_rec_save(os.path.join(self.raw_dir, "ASAPDataset"), "note_alignment.tsv", self.save_dir)
        self.graphs = Parallel(n_jobs=self.nprocs)(
            delayed(gcreate)(
                sfn, pfn, afn, save_name, gname) for (
                sfn, pfn, afn, save_name, gname) in tqdm(load_data, position=0, leave=True))


    def has_cache(self):
        if os.path.exists(self.save_path):
            return True
        return False

    def save(self):
        """save the graph list and the labels"""
        pass

    def load(self):
        self.graphs = list()
        sdir = os.path.join(self.save_dir, self.name)
        for foldir in os.listdir(sdir):
            composer = foldir.split("_")[0]
            for fn in os.listdir(os.path.join(sdir, foldir)):
                path = os.path.join(sdir, foldir, fn)
                graph = load_score_graph(path, fn)
                setattr(graph, "composer", composer)
                setattr(graph, "performer", fn)
                self.graphs.append(graph)

    def __getitem__(self, idx):
        return [[self.graphs[i].x, self.graphs[i].edge_index, self.graphs[i].y, self.graphs[i].mask, self.graphs[i].info["sfn"]] for i in idx]

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


class ASAPGraphAlignmentDataset(StrutturaDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True):
        self.asap_dataset = ASAPDataset(raw_dir=raw_dir)
        if verbose:
            print("Loaded Asap Dataset Successfully, now processing...")
        self.graphs = list()
        super(ASAPGraphAlignmentDataset, self).__init__(
            name="ASAPGraphAlignmentDataset",
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

        def asap_rec_save(par_dir, doc_type, save_dir, result=[]):
            for cdir in os.listdir(par_dir):
                path = os.path.join(par_dir, cdir)
                if os.path.isdir(path):
                    result = asap_rec_save(path, doc_type, save_dir, result)
                else:
                    if path.endswith(doc_type):
                        pdir = os.path.dirname(par_dir)
                        try:
                            if os.path.dirname(os.path.dirname(pdir)) != "ASAPDataset":
                                name = os.path.basename(
                                    os.path.dirname(os.path.dirname(pdir))) + "_" + os.path.basename(
                                    os.path.dirname(pdir)) + "_" + os.path.basename(pdir)
                            else:
                                name = os.path.basename(
                                    os.path.basename(os.path.dirname(pdir)) + "_" + os.path.basename(pdir))
                            save_name = os.path.basename(par_dir).split("_note_alignments")[0]
                            if not os.path.exists(os.path.join(path, name, save_name)):
                                sfn = os.path.join(pdir, "xml_score.musicxml")
                                pfn = os.path.join(pdir, save_name + ".mid")
                                afn = path
                                g = agraph_from_part(sfn=sfn, pfn=pfn, afn=afn, name=save_name)
                                g.save(os.path.join(self.save_dir, self.name, name))
                                result.append(g)
                        except:
                            print("Graph Creation failed on {}".format(path))
            return result

        self.graphs = asap_rec_save(os.path.join(self.raw_dir, "ASAPDataset"), "note_alignment.tsv", self.save_dir)

    def load(self):
        self.graphs = list()
        sdir = os.path.join(self.save_dir, self.name)
        for foldir in os.listdir(sdir):
            composer = foldir.split("_")[0]
            for fn in os.listdir(os.path.join(sdir, foldir)):
                path = os.path.join(sdir, foldir, fn)
                graph = load_score_graph(path, fn)
                setattr(graph, "composer", composer)
                setattr(graph, "performer", fn)
                self.graphs.append(graph)

    def __getitem__(self, idx):
        return [[self.graphs[i].x, self.graphs[i].edge_index, self.graphs[i].y, self.graphs[i].mask[0], self.graphs[i].mask[1]] for i in idx]

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
# class ASAPMatchGraphDataset(MatchGraphDataset):
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
#         dataset_base = ASAPDataset(raw_dir=raw_dir)
#         super(ASAPMatchGraphDataset, self).__init__(
#             dataset_base=dataset_base,
#             raw_dir=raw_dir,
#             force_reload=force_reload,
#             verbose=verbose,
#             nprocs=nprocs,
#             **kwargs)
