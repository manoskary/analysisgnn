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
import shutil
from typing import Union, List, Tuple


class OpenStringQuartets(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True):
        url = "https://github.com/OpenScore/StringQuartets.git"
        self.scores = list()
        self.composers = list()
        self.score_names = list()
        self.performances = list()
        self.performer_names = list()
        self.match = list()
        self.joint_ps = list()
        self.problematic_pieces = [
            "Mozart,_Wolfgang_Amadeus-String_Quartet_No.4_in_C_major,_K.157",
            "Saint-Georges,_Joseph_Bologne-String_Quartet_in_E-flat_major,_Op.1_No.2",
            "Smetana,_Bedřich-String_Quartet_No.1", "Brahms,_Johannes-String_Quartet_No.3,_Op.67",
            "Saint-Georges,_Joseph_Bologne-String_Quartet_in_C_major,_Op.1_No.1",
            "Wolf,_Hugo-String_Quartet"
        ]
        super(OpenStringQuartets, self).__init__(
            name="OpenStringQuartets",
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
        for root, dirs, files in os.walk(os.path.join(self.save_path, "scores")):
            for fn in files:
                name = "-".join(os.path.relpath(root, os.path.join(self.raw_path, "scores")).split(os.sep))
                if fn.endswith(".mscx") and name not in self.problematic_pieces:
                    self.scores.append(os.path.join(root, fn))
                    # The direct subfolder of the raw_dir is the name of the composer
                    self.composers.append(os.path.relpath(root, os.path.join(self.raw_path, "scores")).split(os.sep)[0])
                    self.score_names.append("-".join(os.path.relpath(root, os.path.join(self.raw_path, "scores")).split(os.sep)))

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True

        return False


class OpenLieder(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True):
        url = "https://github.com/OpenScore/Lieder.git"
        self.scores = list()
        self.composers = list()
        self.score_names = list()
        self.performances = list()
        self.performer_names = list()
        self.match = list()
        self.joint_ps = list()
        self.problematic_pieces = [
            "Faltis,_Evelyn-Lieder_fernen_Gedenkens,_Op._posth-1_Unklarheit",
            "Grandval,_Clémence_de-_-L’absence",
            "Grandval,_Clémence_de-_-Les_lucioles",
            "Faltis,_Evelyn-Lieder_fernen_Gedenkens,_Op._posth-4_Heimkehr",
            'Faltis,_Evelyn-Lieder_fernen_Gedenkens,_Op._posth-1_Unklarheit', 'Grandval,_Clémence_de-_-L’absence',
            'Grandval,_Clémence_de-_-Les_lucioles', 'Faltis,_Evelyn-Lieder_fernen_Gedenkens,_Op._posth-4_Heimkehr',
            'Grandval,_Clémence_de-6_Nouvelles_mélodies-5_Rappelle-toi!',
            'Mayer,_Emilie-3_Lieder,_Op.7-3_Wenn_der_Abendstern_die_Rosen',
            'Guest,_Jane_Mary-_-Marion,_or_Will_ye_gang_to_the_burn_side', 'Jaëll,_Marie-Les_Orientales-1_Rêverie',
            'Jaëll,_Marie-Les_Orientales-2_Nourmahal-la-Rousse', 'Jaëll,_Marie-La_mer-2_Causeries_de_vagues',
            'Haydn,_Joseph-10_Canzonets-08_O_tuneful_Voice,_Hob.XXVIa42',
            'Cornelius,_Peter-Liebeslieder,_Op.4-1_In_Lust_und_Schmerzen',
            'Cornelius,_Peter-Liebeslieder,_Op.4-2_Komm,_wir_wandeln_zusammen',
            'Holmès,_Augusta_Mary_Anne-Les_Sept_Ivresses-1_L’Amour',
            'Boulanger,_Lili-Clairières_dans_le_ciel-11_Par_ce_que_j’ai_souffert',
            'Boulanger,_Lili-Clairières_dans_le_ciel-03_Parfois,_je_suis_triste',
            'Boulanger,_Lili-Clairières_dans_le_ciel-09_Les_lilas_qui_avaient_fleuri',
            'Webern,_Anton-5_Lieder_aus_“Der_siebente_Ring”,_Op.3-1_Dies_ist_ein_Lied_für_dich_allein',
            'Kralik,_Mathilde-Blumenlieder-5_Rosen', 'Lang,_Josephine-2_Lieder,_Op.28-1_Traumbild',
            'Lang,_Josephine-6_Lieder,_Op.10-2_Mignons_Klage',
            'Schubert,_Franz-Schwanengesang,_D.957-02_Kriegers_Ahnung', 'Munktell,_Helena-10_Songs-06_Fascination',
            'Corder,_Frederick-_-O_Sun,_That_Wakenest', 'Fauré,_Gabriel-3_Songs,_Op.7-1_Après_un_rêve',
            'Faisst,_Clara_Mathilda-4_Lieder,_Op._11-4_Dulde,_gedulde_dich_fein',
            'Faisst,_Clara_Mathilda-4_Lieder,_Op.18-4_Neue_Liebe',
            'Faisst,_Clara_Mathilda-2_Lieder,_Op.8-1_Die_Insel_der_Vergessenheit',
            'Kinkel,_Johanna-6_Lieder,_Op.15-1_Römische_Nacht',
            'Kinkel,_Johanna-6_Lieder,_Op.15-5_Rette_Vater,_dein_geliebtes_Kind!',
            'Debussy,_Claude-Cinq_Poëmes_de_Baudelaire-5_La_Mort_des_Amants',
            'Debussy,_Claude-Cinq_Poëmes_de_Baudelaire-1_Le_Balcon',
            'Debussy,_Claude-Cinq_Poëmes_de_Baudelaire-2_Harmonie_du_Soir',
            'Chaminade,_Cécile-_-L’allée_d’émeraude_et_d’or', 'Chaminade,_Cécile-_-Au_pays_bleu',
            'Brahms,_Johannes-9_Lieder_und_Gesänge,_Op.32-8_So_stehn_wir,_ich_und_meine_Weide',
            'Brahms,_Johannes-9_Lieder_und_Gesänge,_Op.32-2_Nicht_mehr_zu_dir_zu_gehen',
            'Boulton,_Harold-12_New_Songs-10_For_Ever_Mine', 'Lehmann,_Liza-Bird_Songs-4_The_Wren',
            'Lehmann,_Liza-5_Little_Love_Songs-2_Along_the_sunny_lane',
            'Lehmann,_Liza-Songs_of_Love_and_Spring-05_Disturb_It_Not',
            'Lehmann,_Liza-Songs_of_Love_and_Spring-09_My_Secret',
            'Coleridge-Taylor,_Samuel-6_Sorrow_Songs,_Op.57-1_Oh_what_comes_over_the_Sea',
            'Mahler,_Gustav-Kindertotenlieder-5_In_diesem_Wetter'
        ]
        super(OpenLieder, self).__init__(
            name="OpenLieder",
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
        for root, dirs, files in os.walk(os.path.join(self.save_path, "scores")):
            for fn in files:
                name = "-".join(os.path.relpath(root, os.path.join(self.raw_path, "scores")).split(os.sep))
                if fn.endswith(".mscx") and name not in self.problematic_pieces:
                    self.scores.append(os.path.join(root, fn))
                    # The direct subfolder of the raw_dir is the name of the composer
                    self.composers.append(os.path.relpath(root, os.path.join(self.raw_path, "scores")).split(os.sep)[0])
                    self.score_names.append(name)
        #             try:
        #                 with warnings.catch_warnings():
        #                     warnings.simplefilter("ignore")
        #                     score = pt.load_score(os.path.join(root, fn))
        #             except:
        #                 print("Error loading {}".format(os.path.join(root, fn)))
        #                 self.problematic_pieces.append(name)
        # print(self.problematic_pieces)

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True

        return False


class OpenStringQuartetsGraphDataset(InMemoryDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, num_workers=1):
        self.base_dataset = OpenStringQuartets()
        root = os.path.join(get_download_dir(), "OpenStringQuartetsGraphDataset") if raw_dir is None else os.path.join(
            raw_dir, "OpenStringQuartetsGraphDataset")
        self.verbose = verbose
        self.force_reload = force_reload
        self.num_workers = num_workers
        transform, pre_filter, pre_transform = None, None, None
        self.prob_pieces = [
            "Mozart,_Wolfgang_Amadeus-String_Quartet_No.4_in_C_major,_K.157",
            "Saint-Georges,_Joseph_Bologne-String_Quartet_in_E-flat_major,_Op.1_No.2",
            "Smetana,_Bedřich-String_Quartet_No.1", "Brahms,_Johannes-String_Quartet_No.3,_Op.67",
            "Saint-Georges,_Joseph_Bologne-String_Quartet_in_C_major,_Op.1_No.1",
            "Wolf,_Hugo-String_Quartet"
        ]
        self.test_pieces = []
        super(OpenStringQuartetsGraphDataset, self).__init__(root, transform, pre_filter)
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


class OpenLiederGraphDataset(InMemoryDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, num_workers=1):
        self.base_dataset = OpenLieder()
        root = os.path.join(get_download_dir(), "OpenLiederGraphDataset") if raw_dir is None else os.path.join(
            raw_dir, "OpenLiederGraphDataset")
        self.verbose = verbose
        self.force_reload = force_reload
        self.num_workers = num_workers
        transform, pre_filter, pre_transform = None, None, None
        # not all notes have a measure error
        self.prob_pieces = [
            'Reichardt,_Louise-12_Gesänge-04_Vaters_Klage',
            'pl-cz--d-iii-31_elsner-jozef--juravit-dominus',
            'Cornelius,_Peter-6_Lieder,_Op.5-1_Botschaft',
            'Holmès,_Augusta_Mary_Anne-Mélodies_pour_piano_et_chant-10_Kypris_berceuse',
            'Holmès,_Augusta_Mary_Anne-20_Mélodies-01_Chanson_lointaine',
            'Jaëll,_Marie-Les_Orientales-3_Clair_de_lune',
            'Jaëll,_Marie-4_Mélodies-3_Les_petits_oiseaux',
            'Jaëll,_Marie-Les_Orientales-4_Les_tronçons_du_serpent',
            'Holmès,_Augusta_Mary_Anne-Mélodies_pour_piano_et_chant-19_Un_rêve,\xa0à_2_voix',
            'Bizet,_Georges-20_Mélodies,_Op.21-19_L’esprit_saint',
            'Bizet,_Georges-20_Mélodies,_Op.21-18_Je_n’en_dirai_rien!',
            'Bizet,_Georges-20_Mélodies,_Op.21-02_Le_matin',
            'Bizet,_Georges-20_Mélodies,_Op.21-03_Vieille_chanson',
            'Bizet,_Georges-20_Mélodies,_Op.21-09_Pastorale',
            'Le_Beau,_Luise_Adolpha-2_Duette,_Op.6-1_Frühlingsanfang',
            'Bizet,_Georges-20_Mélodies,_Op.21-17_Chant_d’amour!',
            'Schumann,_Robert-Lieder_und_Gesänge,_Vol.IV,_Op.96-1_Nachtlied',
            'Schumann,_Robert-5_Lieder_und_Gesänge,_Op.127-5_Schlusslied_des_Narren',
            'Schumann,_Robert-5_Lieder_und_Gesänge,_Op.127-1_Sängers_Trost',
            'Schumann,_Robert-Lieder_und_Gesänge,_Vol.IV,_Op.96-2_Schneeglöckchen',
            'Somervell,_Arthur-A_Shropshire_Lad-01_Loveliest_of_Trees,_the_Cherry_now',
            'Somervell,_Arthur-A_Shropshire_Lad-08_Think_no_more,_Lad,_laugh,_be_jolly',
            'Strauss,_Richard-4_Lieder,_Op.27-3_Heimliche_Aufforderung',
            'Debussy,_Claude-Trois_Chansons_de_Bilitis-1_La_flûte_de_Pan',
            'Debussy,_Claude-Trois_Chansons_de_Bilitis-2_La_Chevelure',
            'Debussy,_Claude-Cinq_Poëmes_de_Baudelaire-4_Recueillement',
            'Debussy,_Claude-Cinq_Poëmes_de_Baudelaire-3_Le_Jet_d’Eau',
            'German,_Edward-3_Spring_Songs-1_All_the_World_Awakes_Today'
        ]
        self.test_pieces = []
        super(OpenLiederGraphDataset, self).__init__(root, transform, pre_filter)
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