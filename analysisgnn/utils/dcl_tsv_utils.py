import json
import os
import warnings
from ast import literal_eval
from enum import Enum
from fractions import Fraction
from functools import cache
from typing import Optional, List
from urllib.parse import urlparse

import graphmuse as gm
import numpy as np
import pandas as pd
import requests
import torch

from analysisgnn.descriptors import select_features
from analysisgnn.utils import EMPIRICAL_TONE_FUNCTIONS, EMPIRICAL_KEYS
from analysisgnn.utils.music import PitchEncoder, KeySignatureEncoder
from analysisgnn.utils.chord_representations import encode_one_hot, LocalKey50, TonicizedKey50, ChordQuality15, \
    ChordRoot38, \
    Inversion4, PrimaryDegree22, SecondaryDegree22, Bass38, HarmonicRhythm2, RomanNumeral76, SimpleRomanNumeral185, \
    PitchClassSet94, NoteDegree49
from analysisgnn.utils.globals import INTERVAL_TO_SEMITONES
from analysisgnn.utils.music import CadenceEncoder


def safe_fraction(s: str) -> Fraction | str:
    try:
        return Fraction(s)
    except Exception:
        return s


def safe_literal_eval(s: str):
    try:
        return literal_eval(s)
    except Exception:
        return s

def create_spec_file(
        specs_csv: str,
        **replace_dtypes
) -> pd.DataFrame:
    """

    Args:
        specs_csv:
            Path to a CSV file where the first column contains the column names of the pitch array
            to be loaded and a column "dtype" containing the corresponding dtypes as output by
            pd.DataFrame.dtypes
        **replace_dtypes: Keyword arguments can be used to overwrite the dtypes from the CSV.
    """
    loaded_specs = pd.read_csv(specs_csv, index_col=0)
    converters = dict(
        a_pcset=safe_literal_eval,
        a_pitchNames=safe_literal_eval,
        chord_tones=safe_literal_eval,
        added_tones=safe_literal_eval,
        duration=safe_fraction,
        mn_onset=safe_fraction,
        quarterbeats_playthrough=safe_fraction,
        s_duration_frac=safe_fraction,
        s_offset_frac=safe_fraction,
    )
    dtype_dict = {
        col: dtype
        for col, dtype in loaded_specs.dtype.replace(replace_dtypes).items()
        if col not in converters
    }
    return dtype_dict, converters


def load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_labeled_pitch_array(
        spec_file: dict | str,
        pitch_array_tsv: str,
        converters: dict = None,
        dropna_subset: Optional[str | List[str]] = None,
) -> pd.DataFrame:
    """

    Args:
        spec_file: A {column -> dtype} mapping or a path to a JSON file containing one.
        pitch_array_tsv: path
        dropna: bool

    """
    specs = load_json_file(spec_file) if isinstance(spec_file, str) else spec_file
    result = pd.read_csv(pitch_array_tsv, sep="\t", dtype=specs, converters=converters)
    return result.dropna(subset=dropna_subset) if dropna_subset else result


def create_graph_from_df(
        path,
        spec_file: Optional[str | dict] = None,
        pitch_encoder=PitchEncoder(),
        key_signature_encoder=KeySignatureEncoder(),
        do_labels=True,
        drop_na=True,
        dlc=True,
        converters=None,
        interval="P1",
        verbose: bool = False,
        feature_type="voice"
):

    if drop_na:
        if dlc:
            # currently, DLC contains labels without notes (drop missing 'tpc') and
            # notes without valid labels will be dealt with using the boolean column valid_chord_label
            subset = ["tpc"]
        else:
            subset = "s_note"
    else:
        subset = None

    if verbose:
        print(
            f"Loading pitch array from {path!r}, using {len(converters)} converters, "
            f"and the settings {drop_na} and {subset}."
            )

    df = load_labeled_pitch_array(
        spec_file=spec_file, pitch_array_tsv=path, converters=converters, dropna_subset=subset
        )
    # make the note_array
    df["onset_beat"] = df["continuous_beats"] if "continuous_beats" in df.columns else df["onset_beat"]
    df["ts_beats"] = df["ts_beats"].astype(int)
    df["is_downbeat"] = df["onset_beat"] % 1 == 0
    diff_onset_beat = np.diff(df["onset_beat"].unique())
    diff_onset_div = np.diff(df["onset_div"].unique())
    # assert len(diff_onset_beat) == len(diff_onset_div)
    # calculate how many divs per beat
    divs_per_beat = diff_onset_div[0] / diff_onset_beat[0]
    assert not np.isclose(diff_onset_beat[0], 0), "diff_onset_beat is 0"
    # if divs_per_beat is not constant, raise an error
    df["duration_beat"] = df["duration_div"] / divs_per_beat
    if "step" not in df.columns:
        df["step"] = df["s_step"]
        df["alter"] = df["s_alter"]
    if "pitch" not in df.columns:
        df["pitch"] = df["s_midi"]
    if "is_note_onset" not in df.columns:
        df["is_note_onset"] = df["s_isOnset"]
    if "staff" not in df.columns:
        staffs = df["s_part_id"].unique()
        enum_staff = {staff: i for i, staff in enumerate(staffs)}
        df["staff"] = df["s_part_id"].map(enum_staff)
    if "voice" not in df.columns:
        df["voice"] = df["s_voice_id"].astype(int)

    df["pitch"] = (df["pitch"] + INTERVAL_TO_SEMITONES[interval]) % 128
    note_array = df[["onset_div", "duration_div", "pitch", "is_note_onset", "step", "alter", "onset_beat", "ts_beats", "ts_beat_type", "staff", "voice", "duration_beat", "is_downbeat", "ks_fifths"]]
    # # create structure array and select dtype
    # note_array = note_array.to_records(index=False, column_dtypes={"onset_div": np.int32, "duration_div": np.int32, "pitch": np.int32, "is_note_onset": bool, "step": str, "alter": np.int32, "onset_beat": np.float32, "ts_beats": np.int32, "ts_beat_type": np.int32, "staff": np.int32, "voice": np.int32, "ks_fifths": np.int32, "duration_beat": np.float32})
    note_array = note_array.to_records(index=False)
    features = select_features(note_array, feature_type)
    mc_playthrough = df["mn_playthrough"].to_numpy() if "mn_playthrough" in df.columns else df["measureNumberWithSuffix"].to_numpy()
    # find where mc_playthrough changes
    change_indices = np.where(mc_playthrough[:-1] != mc_playthrough[1:])[0]
    change_indices = np.r_[0, change_indices + 1]
    measure_onset_div = note_array["onset_div"][change_indices]
    offsets = df["onset_div"].to_numpy() + df["duration_div"].to_numpy()
    last_offset = offsets[-1]
    measure_offset_div = np.r_[offsets[change_indices[1:]], last_offset]
    # measures is tuples of (onset_div, offset_div)
    measures = np.vstack((measure_onset_div, measure_offset_div)).T
    # graph = gm.create_score_graph(features, note_array, measures=measures, add_beats=True)  # Commented out due to missing graphmuse
    
    # Temporary replacement: try to use our local hetero_graph_from_note_array
    try:
        from analysisgnn.utils.hgraph import hetero_graph_from_note_array
        graph = hetero_graph_from_note_array(note_array)
    except Exception:
        # If that fails, return None - this will need to be properly implemented later
        return None
        
    if graph is None:
        return None
    pitch_spelling = pitch_encoder.encode(note_array).astype(int)
    key_signature = key_signature_encoder.encode(note_array).astype(int)
    if interval != "P1":
        pitch_spelling = pitch_encoder.transpose(pitch_spelling, interval)
        key_signature = key_signature_encoder.transpose(key_signature, interval)
    graph["note"].pitch_spelling = torch.from_numpy(pitch_spelling).long()
    graph["note"].key_signature = torch.from_numpy(key_signature).long()
    graph["note"].is_note_onset = torch.from_numpy(note_array["is_note_onset"].astype(bool)).long()
    graph["name"] = f"{os.path.splitext(os.path.basename(path))[0]}_{interval}"
    graph["collection"] = os.path.basename(os.path.dirname(path))
    graph["transposition"] = interval
    if do_labels:
        if dlc:
            errors = ErrorHandling.WARN if verbose else ErrorHandling.IGNORE
            labels = create_labels_dlc(df, interval=interval, errors=errors)
        else:
            labels = create_labels(df, interval=interval)
        for key, value in labels.items():
            graph["note"][key] = torch.from_numpy(value).long()
    return graph


def list_files_in_folder(repo_url, folder_path, branch='v100_notes'):
    # Parse the URL to extract the repository owner and name
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')
    repo_owner = path_parts[0]
    repo_name = path_parts[1]

    # Construct the API URL
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{folder_path}?ref={branch}"
    response = requests.get(api_url)

    if response.status_code == 200:
        contents = response.json()
        files = [item['download_url'] for item in contents if item['type'] == 'file']
        return files
    else:
        raise Exception(f"Failed to fetch folder contents: {response.status_code}")


def list_folders_in_folder(repo_url, folder_path, branch='v100_notes'):
    # Parse the URL to extract the repository owner and name
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')
    repo_owner = path_parts[0]
    repo_name = path_parts[1]

    # Construct the API URL
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{folder_path}?ref={branch}"
    response = requests.get(api_url)

    if response.status_code == 200:
        contents = response.json()
        folder_paths = [f"{folder_path}/{item['name']}" for item in contents if item['type'] == 'dir']
        return folder_paths
    else:
        raise Exception(f"Failed to fetch folder contents: {response.status_code}")


class ErrorHandling(str, Enum):
    RAISE = "raise"
    """Raise the exception."""
    IGNORE = "ignore"
    """Return None"""
    WARN = "warn"
    """Like IGNORE but throws a warning containing the exception."""

NOTE_NAMES = ("F", "C", "G", "D", "A", "E", "B")


@cache
def fifths_to_name(
        fifths: int,
        lowest: int = -15,
        highest: int = 19,
        flat_sign: str = "-",
        errors: ErrorHandling = ErrorHandling.RAISE
) -> Optional[str]:
    try:
        fifths = int(fifths)
        assert lowest <= fifths <= highest, f"fifths value needs to be within [{lowest}, {highest}], got {fifths}"
    except Exception as e:
        if errors == ErrorHandling.RAISE:
            raise
        if errors == ErrorHandling.IGNORE:
            return
        if errors == ErrorHandling.WARN:
            warnings.warn(f"Converting {fifths} to a note name failed with {e}")
            return
    accidentals, ix = divmod(int(fifths) + 1, 7)
    acc_str = abs(accidentals) * flat_sign if accidentals < 0 else accidentals * "#"
    return f"{NOTE_NAMES[ix]}{acc_str}"


def process_inversion_from_chord(x):
    # if x is None, return 0
    if pd.isna(x):
        return 0
    elif x == "6" or x == "65":
        return 1
    elif x == "64" or x == "43":
        return 2
    elif x == "42" or x == "2":
        return 3
    else:
        return 0


def alter_to_str(alter):
    if alter == 1:
        return "#"
    elif alter == 2:
        return "##"
    elif alter == -1:
        return "-"
    elif alter == -2:
        return "--"
    else:
        return ""


def is_element_in_list_array(string_array, list_array):
    """
    Check if each element in string_array is in the corresponding list in list_array

    Args:
        string_array: Array of strings
        list_array: Array of lists/tuples containing strings

    Returns:
        Boolean array indicating if each element is in its corresponding list
    """
    result = np.zeros(len(string_array), dtype=int)
    for i, (s, lst) in enumerate(zip(string_array, list_array)):
        if pd.notna(s) and pd.notna(lst):
            if s in lst:
                result[i] = 1
    return result


def create_labels(filtered_df, interval="P1"):

    # Transformation functions
    filtered_df["a_degree2"] = filtered_df["a_degree2"].apply(lambda x: None if pd.isna(x) else x)
    filtered_df["tpc"] = filtered_df["step"] + filtered_df["alter"].apply(alter_to_str)
    tpc_is_bass = (filtered_df["tpc"] == filtered_df["a_bass"]).to_numpy().astype(int)
    tpc_is_root = (filtered_df["tpc"] == filtered_df["a_root"]).to_numpy().astype(int)
    # Create a universal function for checking membership in each corresponding tuple
    ufunc_in = np.frompyfunc(lambda t, p: t in p, 2, 1)
    # Apply the universal function and cast the result to int type
    tpc_is_in_label = ufunc_in(filtered_df["tpc"].to_numpy(), filtered_df["a_pitchNames"].to_numpy()).astype(int)
    # filtered_df.a_localKey = filtered_df.a_localKey.str.replace("-", "b")
    # filtered_df.a_tonicizedKey = filtered_df.a_tonicizedKey.str.replace("-", "b")
    # filtered_df.a_root = filtered_df.a_root.str.replace("-", "b")
    # filtered_df.a_bass = filtered_df.a_bass.str.replace("-", "b")

    # Encode the labels
    localkey = encode_one_hot(filtered_df, LocalKey50, transposition=interval).astype(int)
    tonkey = encode_one_hot(filtered_df, TonicizedKey50, transposition=interval).astype(int)
    quality = encode_one_hot(filtered_df, ChordQuality15, transposition=interval).astype(int)
    root = encode_one_hot(filtered_df, ChordRoot38, transposition=interval).astype(int)
    inversion = encode_one_hot(filtered_df, Inversion4, transposition=interval).astype(int)
    degree1 = encode_one_hot(filtered_df, PrimaryDegree22, transposition=interval).astype(int)
    degree2 = encode_one_hot(filtered_df, SecondaryDegree22, transposition=interval).astype(int)
    bass = encode_one_hot(filtered_df, Bass38, transposition=interval).astype(int)
    hrythm = encode_one_hot(filtered_df, HarmonicRhythm2, transposition=interval).astype(int)
    pcset = encode_one_hot(filtered_df, PitchClassSet94, transposition=interval).astype(int)
    romanNumeral = encode_one_hot(filtered_df, SimpleRomanNumeral185, transposition=interval).astype(int)
    valid_label = filtered_df["valid_chord_label"].to_numpy().astype(bool)
    labels = {
        "localkey": localkey.squeeze(),
        "tonkey": tonkey.squeeze(),
        "quality": quality.squeeze(),
        "root": root.squeeze(),
        "inversion": inversion.squeeze(),
        "degree1": degree1.squeeze(),
        "degree2": degree2.squeeze(),
        "bass": bass.squeeze(),
        "hrythm": hrythm.squeeze(),
        "romanNumeral": romanNumeral.squeeze(),
        "pcset": pcset.squeeze(),
        "tpc_in_label": tpc_is_in_label,
        "tpc_is_root": tpc_is_root,
        "tpc_is_bass": tpc_is_bass,
        "valid_label": valid_label
    }
    return labels


def create_labels_dlc(filtered_df, interval="P1", errors: ErrorHandling = ErrorHandling.RAISE):
    cadence_encoder = CadenceEncoder()
    # step and alter to pitch name
    filtered_df['a_root'] = filtered_df['a_root'].apply(lambda x: EMPIRICAL_TONE_FUNCTIONS[x])
    filtered_df['a_bass'] = filtered_df['a_bass'].apply(lambda x: EMPIRICAL_TONE_FUNCTIONS[x])
    filtered_df['a_localKey'] = filtered_df['a_localKey'].apply(lambda x: EMPIRICAL_KEYS[x])
    filtered_df['a_tonicizedKey'] = filtered_df['a_tonicizedKey'].apply(lambda x: EMPIRICAL_KEYS[x])
    # turn pd.NA to None
    filtered_df = filtered_df.map(lambda x: None if pd.isna(x) else x)
    # Encode the labels
    localkey = encode_one_hot(filtered_df, LocalKey50, transposition=interval).astype(int)
    tonkey = encode_one_hot(filtered_df, TonicizedKey50, transposition=interval).astype(int)
    quality = encode_one_hot(filtered_df, ChordQuality15, transposition=interval).astype(int)
    root = encode_one_hot(filtered_df, ChordRoot38, transposition=interval).astype(int)
    inversion = encode_one_hot(filtered_df, Inversion4, transposition=interval).astype(int)
    degree1 = encode_one_hot(filtered_df, PrimaryDegree22, transposition=interval).astype(int)
    degree2 = encode_one_hot(filtered_df, SecondaryDegree22, transposition=interval).astype(int)
    bass = encode_one_hot(filtered_df, Bass38, transposition=interval).astype(int)
    hrythm = encode_one_hot(filtered_df, HarmonicRhythm2, transposition=interval).astype(int)
    romanNumeral = encode_one_hot(filtered_df, SimpleRomanNumeral185, transposition=interval).astype(int)
    note_degree = encode_one_hot(filtered_df, NoteDegree49, transposition=interval).astype(int)

    # Other labels
    metrical_strength = filtered_df["downbeat"].to_numpy()
    staff = filtered_df["staff"].to_numpy().astype(int)
    voice = filtered_df["voice"].to_numpy().astype(int)
    section = filtered_df["section_start"].to_numpy().astype(int)
    phrase = filtered_df["a_phraseend"].to_numpy().astype(int)
    tpc_in_label = filtered_df["tpc_is_in_label"].to_numpy().astype(int)
    tpc_is_root = filtered_df["tpc_is_root"].to_numpy().astype(int)
    tpc_is_bass = filtered_df["tpc_is_bass"].to_numpy().astype(int)
    cadence = filtered_df["cadence_type"].apply(lambda x: cadence_encoder.encode_from_text(x) if pd.notna(x) else 0).to_numpy().astype(int)
    organ_point = filtered_df["pedal"].apply(lambda x: 1 if pd.notna(x) else 0).to_numpy().astype(int)
    valid_label = filtered_df["valid_chord_label"].to_numpy().astype(bool)
    valid_cadence_label = filtered_df["valid_cadence_label"].to_numpy().astype(bool)
    valid_phrase_label = filtered_df["valid_phrase_label"].to_numpy().astype(bool)
    valid_organ_point_label = filtered_df["valid_pedal_point_label"].to_numpy().astype(bool)
    valid_section_start_label = filtered_df["valid_section_start_label"].to_numpy().astype(bool)
    downbeat = filtered_df["downbeat"].to_numpy().astype(int)
    assert np.all(downbeat < 45), f"downbeat is not in range 45, got {downbeat}"

    labels = {
        "localkey": localkey.squeeze(),
        "tonkey": tonkey.squeeze(),
        "quality": quality.squeeze(),
        "root": root.squeeze(),
        "inversion": inversion.squeeze(),
        "degree1": degree1.squeeze(),
        "degree2": degree2.squeeze(),
        "bass": bass.squeeze(),
        "hrythm": hrythm.squeeze(),
        "romanNumeral": romanNumeral.squeeze(),
        "metrical_strength": metrical_strength,
        "downbeat": downbeat,
        # "staff": staff,
        # "voice": voice,
        "note_degree": note_degree.squeeze(),
        "section": section,
        "phrase": phrase,
        "tpc_in_label": tpc_in_label,
        "tpc_is_root": tpc_is_root,
        "tpc_is_bass": tpc_is_bass,
        "cadence": cadence,
        "pedal": organ_point,
        "valid_label": valid_label,
        "valid_cadence_label": valid_cadence_label,
        "valid_phrase_label": valid_phrase_label,
        "valid_organ_point_label": valid_organ_point_label,
        "valid_section_start_label": valid_section_start_label
    }
    return labels




