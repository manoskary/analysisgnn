import os
import shutil
from typing import List, Union, Tuple

import torch
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from analysisgnn.utils.dcl_tsv_utils import create_graph_from_df
from analysisgnn.data.dataset import BuiltinDataset
from analysisgnn.data.dataset import get_download_dir


def make_dlc_nickname(c_name, p_name):
    nickname = f"{c_name}_{p_name}"
    return nickname

class DLCDataset(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True):
        url = "https://github.com/johentsch/dilemmadata.git"
        super(DLCDataset, self).__init__(
            name="dilemmadata",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            is_zip=False,
            branch="main",
            clone=True
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
        folder_path = "pitch_arrays/DLC"
        path = os.path.join(self.raw_path, folder_path)
        assert os.path.isdir(path), "The directory does not exist maybe you need to checkout the correct branch"
        for folder in os.listdir(path):
            for file in os.listdir(os.path.join(path, folder)):
                if file.endswith(".tsv"):
                    self.scores.append(os.path.join(path, folder, file))
                    self.collections.append(folder)
                    self.composers.append(folder.split("_")[0])

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True
        return False


class DLCGraphDataset(InMemoryDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, num_workers=1, transpose=False, name=None):
        name = "DLCGraphDataset" if name is None else name
        self.base_dataset = DLCDataset(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)
        root = os.path.join(get_download_dir(), name) if raw_dir is None else os.path.join(
            raw_dir, name)
        self.verbose = verbose
        self.force_reload = force_reload
        self.num_workers = num_workers
        self.transpose = transpose
        self.all_intervals = ["P1", "m2", "M2", "m3", "M3", "P4", "A4", "P5", "m6", "M6", "m7", "M7"]
        transform, pre_filter, pre_transform = None, None, None
        # not all notes have a measure error
        self.prob_pieces = [
            ## included in AugmentedNet test set
            'ABC_n01op18-1_01',
            'ABC_n01op18-1_03',
            'ABC_n06op18-6_03',
            'ABC_n07op59-1_01',
            'ABC_n08op59-2_03',
            'ABC_n10op74_03',
            'ABC_n10op74_04',
            'ABC_n11op95_03',
            'ABC_n12op127_02',
            'ABC_n16op135_02',
            'beethoven_piano_sonatas_01-1',
            'beethoven_piano_sonatas_07-1',
            'beethoven_piano_sonatas_10-1',
            'beethoven_piano_sonatas_23-1',
            'monteverdi_madrigals_5-04d'
        ]
        self.test_pieces = [
            'bach_en_fr_suites_BWV807_06_Bourree_II',
            'bach_en_fr_suites_BWV807_07_Gigue',
            'bach_en_fr_suites_BWV808_01_Prelude',
            'bach_en_fr_suites_BWV809_05_Menuett_I',
            'bach_en_fr_suites_BWV809_07_Gigue',
            'bach_en_fr_suites_BWV810_02_Allemande',
            'bach_en_fr_suites_BWV810_04_Sarabande',
            'bach_en_fr_suites_BWV810_05_Passepied_I',
            'bach_en_fr_suites_BWV810_06_Passepied_II',
            'bach_en_fr_suites_BWV811_05_Double',
            'bach_en_fr_suites_BWV812_01_Allemande',
            'bach_en_fr_suites_BWV812_04_Menuett_I',
            'bach_en_fr_suites_BWV813_01_Allemande',
            'bach_en_fr_suites_BWV813_02_Courante',
            'bach_en_fr_suites_BWV813_05_Menuett',
            'bach_en_fr_suites_BWV814_04_Gavotte',
            'bach_en_fr_suites_BWV814_06_Trio',
            'bach_en_fr_suites_BWV815_02_Courante',
            'bach_en_fr_suites_BWV815_05_Air',
            'bach_en_fr_suites_BWV815_06_Menuett',
            'bach_en_fr_suites_BWV816_01_Allemande',
            'bach_en_fr_suites_BWV816_03_Sarabande',
            'bach_en_fr_suites_BWV816_04_Gavotte',
            'bach_en_fr_suites_BWV816_07_Gigue',
            'bach_en_fr_suites_BWV817_06_Bourree',
            'bach_en_fr_suites_BWV817_07_Gigue',
            'bach_solo_BWV1009_06_BourréeII',
            'bach_solo_BWV1010_02_Allemande',
            'bach_solo_BWV1010_04_Sarabande',
            'bach_solo_BWV1011_05_GavotteI',
            'bach_solo_BWV1012_01_Prelude',
            'bach_solo_BWV1012_02_Allemande',
            'bach_solo_BWV1012_07_Gique',
            'bach_solo_BWV1013_01_Allemande',
            'bach_solo_BWV1013_02_Corrente',
            'bach_solo_BWV1013_03_Sarabande',
            'bartok_bagatelles_op06n01',
            'bartok_bagatelles_op06n02',
            'bartok_bagatelles_op06n03',
            'bartok_bagatelles_op06n10',
            'beethoven_piano_sonatas_01-3',
            'beethoven_piano_sonatas_03-4',
            'beethoven_piano_sonatas_05-3',
            'beethoven_piano_sonatas_06-2',
            'beethoven_piano_sonatas_08-1',
            'beethoven_piano_sonatas_08-3',
            'beethoven_piano_sonatas_09-1',
            'beethoven_piano_sonatas_09-2',
            'beethoven_piano_sonatas_09-3',
            'beethoven_piano_sonatas_16-2',
            'beethoven_piano_sonatas_19-1',
            'beethoven_piano_sonatas_20-2',
            'beethoven_piano_sonatas_21-1',
            'beethoven_piano_sonatas_21-3',
            'beethoven_piano_sonatas_23-2',
            'beethoven_piano_sonatas_24-2',
            'beethoven_piano_sonatas_30-2',
            'beethoven_piano_sonatas_31-2',
            'c_schumann_lieder_op13no2 Sie liebten sich beide',
            'c_schumann_lieder_op23no1 Was weinst du Blumlein',
            'c_schumann_lieder_op23no5 Das ist ein Tag der klingen mag',
            'chopin_mazurkas_BI105-4op30-4',
            'chopin_mazurkas_BI115-2op33-2',
            'chopin_mazurkas_BI126-3op41-1',
            'chopin_mazurkas_BI145-1op50-1',
            'chopin_mazurkas_BI145-3op50-3',
            'chopin_mazurkas_BI153-2op56-2',
            'chopin_mazurkas_BI153-3op56-3',
            'chopin_mazurkas_BI157-2op59-2',
            'chopin_mazurkas_BI168op68-4',
            'chopin_mazurkas_BI60-2op06-2',
            'chopin_mazurkas_BI61-2op07-2',
            'chopin_mazurkas_BI61-4op07-4',
            'chopin_mazurkas_BI77-1op17-1',
            'chopin_mazurkas_BI77-3op17-3',
            'corelli_op01n03d',
            'corelli_op01n04d',
            'corelli_op01n05a',
            'corelli_op01n06a',
            'corelli_op01n07c',
            'corelli_op01n08c',
            'corelli_op01n09c',
            'corelli_op01n10b',
            'corelli_op01n11c',
            'corelli_op01n12a',
            'corelli_op01n12d',
            'corelli_op03n01b',
            'corelli_op03n03a',
            'corelli_op03n03c',
            'corelli_op03n04b',
            'corelli_op03n07b',
            'corelli_op03n07d',
            'corelli_op03n08b',
            'corelli_op03n08d',
            'corelli_op03n09a',
            'corelli_op03n09b',
            'corelli_op03n09c',
            'corelli_op03n09d',
            'corelli_op03n10a',
            'corelli_op03n10b',
            'corelli_op03n10c',
            'corelli_op03n12a',
            'corelli_op03n12e',
            'corelli_op04n01d',
            'corelli_op04n02c',
            'corelli_op04n03d',
            'corelli_op04n05b',
            'corelli_op04n05c',
            'corelli_op04n06b',
            'corelli_op04n06d',
            'corelli_op04n06g',
            'corelli_op04n07d',
            'corelli_op04n10a',
            'corelli_op04n10b',
            'corelli_op04n11a',
            'corelli_op04n11b',
            'corelli_op04n11c',
            'couperin_clavecin_00_allemande',
            'couperin_clavecin_01_premier_prelude',
            'couperin_concerts_c03n05_gavotte',
            'couperin_concerts_c03n07_musette_2',
            'couperin_concerts_c05n01_prelude',
            'couperin_concerts_c05n02_allemande',
            'couperin_concerts_c05n05_musete',
            'couperin_concerts_c06n02_allemande',
            'couperin_concerts_c06n03_sarabande',
            'couperin_concerts_c07n01_grave',
            'couperin_concerts_c07n02_allemande',
            'couperin_concerts_c07n04_fuguete',
            'couperin_concerts_c08n02_ritournele',
            'couperin_concerts_c08n05_air_leger',
            'couperin_concerts_c08n09_air_leger',
            'couperin_concerts_c09n02_lenjouement',
            'couperin_concerts_c09n03_graces',
            'couperin_concerts_c09n07_douceur',
            'couperin_concerts_c10n01_gravement',
            'couperin_concerts_c11n01_majestueusement',
            'couperin_concerts_c11n02_allemande',
            'couperin_concerts_c11n07_gigue',
            'couperin_concerts_c14n04_fuguete',
            'couperin_concerts_parnasse_02',
            'couperin_concerts_parnasse_04',
            'couperin_concerts_parnasse_05',
            'cpe_bach_keyboard_wq114n07',
            'cpe_bach_keyboard_wq119n07',
            'cpe_bach_keyboard_wq50n02c',
            'cpe_bach_keyboard_wq50n03c',
            'cpe_bach_keyboard_wq50n04c',
            'cpe_bach_keyboard_wq55n01c',
            'cpe_bach_keyboard_wq55n03a',
            'cpe_bach_keyboard_wq55n03b',
            'cpe_bach_keyboard_wq55n04a',
            'cpe_bach_keyboard_wq55n04c',
            'cpe_bach_keyboard_wq55n05c',
            'cpe_bach_keyboard_wq55n06b',
            'cpe_bach_keyboard_wq55n06c',
            'cpe_bach_keyboard_wq56n02b',
            'cpe_bach_keyboard_wq56n03',
            'cpe_bach_keyboard_wq57n02a',
            'cpe_bach_keyboard_wq57n02c',
            'cpe_bach_keyboard_wq57n03',
            'dvorak_silhouettes_op08n01',
            'dvorak_silhouettes_op08n04',
            'dvorak_silhouettes_op08n09',
            'dvorak_silhouettes_op08n10',
            'grieg_lyric_pieces_op12n03',
            'grieg_lyric_pieces_op12n04',
            'grieg_lyric_pieces_op38n01',
            'grieg_lyric_pieces_op38n02',
            'grieg_lyric_pieces_op38n07',
            'grieg_lyric_pieces_op43n01',
            'grieg_lyric_pieces_op47n07',
            'grieg_lyric_pieces_op54n01',
            'grieg_lyric_pieces_op54n04',
            'grieg_lyric_pieces_op54n06',
            'grieg_lyric_pieces_op57n01',
            'grieg_lyric_pieces_op62n02',
            'grieg_lyric_pieces_op62n04',
            'grieg_lyric_pieces_op65n01',
            'grieg_lyric_pieces_op65n02',
            'grieg_lyric_pieces_op65n03',
            'grieg_lyric_pieces_op65n04',
            'grieg_lyric_pieces_op71n04',
            'jc_bach_sonatas_wa02op05no2d_Minore',
            'jc_bach_sonatas_wa04op05no4a_Allegro',
            'jc_bach_sonatas_wa06op05no6a_Grave',
            'jc_bach_sonatas_wa06op05no6b_Allegro_Moderato',
            'jc_bach_sonatas_wa08op17no2a_Allegro',
            'jc_bach_sonatas_wa08op17no2b_Andante',
            'jc_bach_sonatas_wa10op17no4a_Allegro',
            'jc_bach_sonatas_wa12op17no6a_Allegro',
            'liszt_pelerinage_160.05_Orage',
            'liszt_pelerinage_160.06_Vallee_dObermann',
            'liszt_pelerinage_160.09_Les_Cloches_de_Geneve_(Nocturne)',
            'liszt_pelerinage_161.05_Sonetto_104_del_Petrarca',
            'liszt_pelerinage_161.06_Sonetto_123_del_Petrarca',
            'liszt_pelerinage_162.02_Canzone',
            'medtner_tales_op26n01',
            'medtner_tales_op26n02',
            'medtner_tales_op34n04',
            'medtner_tales_op35n01',
            'medtner_tales_op35n04',
            'medtner_tales_op42n03',
            'mozart_piano_sonatas_K280-2',
            'mozart_piano_sonatas_K281-3',
            'mozart_piano_sonatas_K282-1',
            'mozart_piano_sonatas_K309-1',
            'mozart_piano_sonatas_K309-3',
            'mozart_piano_sonatas_K310-1',
            'mozart_piano_sonatas_K310-2',
            'mozart_piano_sonatas_K330-3',
            'mozart_piano_sonatas_K332-1',
            'mozart_piano_sonatas_K333-1',
            'mozart_piano_sonatas_K457-1',
            'mozart_piano_sonatas_K533-1',
            'mozart_piano_sonatas_K545-1',
            'mozart_piano_sonatas_K570-3',
            'mozart_piano_sonatas_K576-2',
            'pleyel_quartets_b307op2n1a',
            'pleyel_quartets_b307op2n1b',
            'scarlatti_sonatas_K006',
            'scarlatti_sonatas_K012',
            'scarlatti_sonatas_K017',
            'scarlatti_sonatas_K023',
            'scarlatti_sonatas_K031',
            'scarlatti_sonatas_K035',
            'scarlatti_sonatas_K036',
            'scarlatti_sonatas_K039',
            'scarlatti_sonatas_K040',
            'scarlatti_sonatas_K050',
            'scarlatti_sonatas_K051',
            'scarlatti_sonatas_K053',
            'scarlatti_sonatas_K054',
            'scarlatti_sonatas_K059',
            'scarlatti_sonatas_K064',
            'scarlatti_sonatas_K068',
            'scarlatti_sonatas_K071',
            'scarlatti_sonatas_K072',
            'scarlatti_sonatas_K097',
            'scarlatti_sonatas_K098',
            'schumann_kinderszenen_n03',
            'schumann_kinderszenen_n09',
            'schumann_kinderszenen_n10',
            'schumann_kinderszenen_n13',
            'tchaikovsky_seasons_op37a03',
            'tchaikovsky_seasons_op37a06',
            'tchaikovsky_seasons_op37a07',
            'tchaikovsky_seasons_op37a11',
            'wf_bach_sonatas_F002_n07b',
            'wf_bach_sonatas_F002_n07c'
        ]
        super(DLCGraphDataset, self).__init__(root, transform, pre_filter)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        fns = []
        if not os.path.exists(self.raw_dir) and not os.path.isdir(self.raw_dir):
            os.makedirs(self.raw_dir)
        for ofp, ofc in zip(self.base_dataset.scores, self.base_dataset.collections):
            fn = make_dlc_nickname(ofc, os.path.basename(ofp))
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
        p_name, _ = os.path.splitext(os.path.basename(raw_path))
        nickname = make_dlc_nickname(collection, p_name)

        if nickname in self.prob_pieces:
            return data_list
        intervals = self.all_intervals if self.transpose and nickname not in self.test_pieces else ["P1"]
        for interval in intervals:
            name = nickname if interval == "P1" else f"{nickname}_{interval}"
            save_fp = os.path.join(self.processed_dir, name + ".pt")
            if os.path.exists(save_fp) and not self.force_reload:
                d = torch.load(save_fp)
                data_list.append(d)
                continue
            try:
                d = create_graph_from_df(
                    raw_path,
                    pitch_encoder=pitch_encoder,
                    do_labels=True,
                    drop_na=True,
                    dlc=True,
                    spec_file=spec_file,
                    converters=converters,
                    verbose=self.verbose,
                    interval=interval,
                    key_signature_encoder=key_signature_encoder
                )
                d["collection"] = collection
                d["name"] = nickname
                d["transposition"] = interval
                d["interval"] = interval
                d["test"] = nickname in self.test_pieces
                torch.save(d, save_fp)
            except Exception as e:
                d = None
                if self.verbose:
                    print("Error processing {} with transposition {}".format(raw_path, interval))

            if d is not None:
                data_list.append(d)

        return data_list

    def process(self):
        from analysisgnn.models.pitch_spelling import PitchEncoder, KeySignatureEncoder
        from analysisgnn.utils.dcl_tsv_utils import create_graph_from_df, create_spec_file
        from tqdm import tqdm
        from tqdm.contrib.concurrent import process_map

        data_list = []
        pitch_encoder = PitchEncoder()
        key_signature_encoder = KeySignatureEncoder()
        replace_dtypes = dict(object="string", int64="Int64")
        spec_file_path = os.path.join(self.base_dataset.raw_path, "processing", "DLC", "dlc_pitch_array_specs.csv")        
        assert os.path.exists(spec_file_path), "The spec file does not exist, or is not in the correct location."
        spec_file, converters = create_spec_file(
            spec_file_path, **replace_dtypes)
        if self.num_workers > 1:
            data = [(raw_path, collection, spec_file, converters, pitch_encoder, key_signature_encoder) for raw_path, collection in zip(self.base_dataset.scores, self.base_dataset.collections)]
            data_list = process_map(self._process_single, data, max_workers=self.num_workers)
            data_list = [d for dl in data_list for d in dl]
        else:
            for raw_path, collection in tqdm(zip(self.base_dataset.scores, self.base_dataset.collections), desc="Processing DLC Dataset"):
                data = (raw_path, collection, spec_file, converters, pitch_encoder, key_signature_encoder)
                d = self._process_single(data)
                data_list.extend(d)

        self.save(data_list, self.processed_paths[0])


class DLCplusGraphDataset(DLCGraphDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, num_workers=1, transpose=False):
        super().__init__(
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            num_workers=num_workers,
            transpose=transpose,
            name="DLCplusGraphDataset"
        )

    def _process_single(self, data):
        raw_path, collection, spec_file, converters, pitch_encoder, key_signature_encoder = data
        data_list = []
        p_name, _ = os.path.splitext(os.path.basename(raw_path))
        nickname = make_dlc_nickname(collection, p_name)
        if nickname in self.prob_pieces:
            return data_list
        intervals = self.all_intervals if self.transpose and nickname not in self.test_pieces else ["P1"]
        for interval in intervals:
            save_fp = os.path.join(self.processed_dir, nickname + ".pt") if interval == "P1" else os.path.join(self.processed_dir, f"{nickname}_{interval}.pt")
            if os.path.exists(save_fp) and not self.force_reload:
                d = torch.load(save_fp)
                data_list.append(d)
                continue
            try:
                d = create_graph_from_df(
                    raw_path,
                    pitch_encoder=pitch_encoder,
                    do_labels=True,
                    drop_na=True,
                    dlc=True,
                    spec_file=spec_file,
                    converters=converters,
                    verbose=self.verbose,
                    interval=interval,
                    key_signature_encoder=key_signature_encoder,
                    feature_type="cadence"
                )
                d["collection"] = collection
                d["name"] = nickname
                d["transposition"] = interval
                d["interval"] = interval
                d["test"] = nickname in self.test_pieces
                torch.save(d, save_fp)
            except Exception as e:
                d = None
                if self.verbose:
                    print("Error processing {} with transposition {}".format(raw_path, interval))

            if d is not None:
                data_list.append(d)

        return data_list