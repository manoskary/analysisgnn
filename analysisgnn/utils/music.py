import numpy as np
import partitura as pt
import torch
import re
from partitura.score import Interval

class PitchEncoder(object):
    def __init__(self):
        self.PITCHES = {
            0: ["C", "B#", "D--"],
            1: ["C#", "B##", "D-"],
            2: ["D", "C##", "E--"],
            3: ["D#", "E-", "F--"],
            4: ["E", "D##", "F-"],
            5: ["F", "E#", "G--"],
            6: ["F#", "E##", "G-"],
            7: ["G", "F##", "A--"],
            8: ["G#", "A-"],
            9: ["A", "G##", "B--"],
            10: ["A#", "B-", "C--"],
            11: ["B", "A##", "C-"],
        }
        self.accepted_pitches = np.array([ii for i in self.PITCHES.values() for ii in i])
        self.KEY_SIGNATURES = list(range(-7, 8))
        self.encode_dim = len(self.accepted_pitches)
        self.num_classes = len(self.accepted_pitches)
        self.classes_ = np.unique(self.accepted_pitches)
        self.transposition_dict = {}

    def rooting_function(self, x):
        if x[1] == 0:
            suffix = ""
        elif x[1] == 1:
            suffix = "#"
        elif x[1] == 2:
            suffix = "##"
        elif x[1] == -1:
            suffix = "-"
        elif x[1] == -2:
            suffix = "--"
        else:
            raise ValueError(f"Alteration {x[1]} is not supported")
        out = x[0] + suffix
        return out

    def encode(self, note_array):
        """
        One-hot encoding of pitch spelling triplets.

        x has to be a partitura note_array
        """
        pitch_spelling = note_array[["step", "alter"]]
        root = self.rooting_function
        y = np.vectorize(root)(pitch_spelling)
        return np.searchsorted(self.classes_, y)

    def decode(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return self.classes_[x]
    
    def decode_to_step_alter(self, x):
        """
        Decode integer labels to pitch spelling triplets.

        Parameters
        ----------
        x: numpy array or torch tensor
            Pitch spelling integer labels.

        Returns
        -------
        out: numpy structured array
            Pitch spelling triplets with 'step' and 'alter' fields.
        """        
        decoded = self.decode(x)
        step = np.array([p[0] for p in decoded])
        alter = np.array([p.count("#") - p.count("-") for p in decoded])
        return np.array(list(zip(step, alter)), dtype=[("step", "U2"), ("alter", int)])

    def transpose(self, x, interval):
        """
        Transpose pitch spelling by an interval.

        Parameters
        ----------
        x : numpy array or torch tensor
            Pitch spelling integer labels.
        interval : partitura.Interval or str
            The interval by which to transpose the pitch spelling.
        """
        to_tensor = False
        device = None
        if isinstance(x, torch.Tensor):
            device = x.device
            tdtype = x.dtype
            x = x.detach().cpu().numpy()
            to_tensor = True
        if isinstance(interval, str):
            # Quality is any of the following: P, M, m, A, d
            quality = re.findall(r"[PMAmd]", interval)[0]
            number = int(re.findall(r"\d+", interval)[0])
            interval = Interval(number, quality)
        interval_name = interval.quality + str(interval.number)
        if interval_name not in self.transposition_dict.keys():
            self.introduce_transposition(interval)
        if not np.all(np.isin(x, self.transposition_dict[interval_name]["accepted_indices"])):
            # if there are pitches that cannot be transposed
            raise ValueError("Some pitches cannot be transposed by the given interval")
        reindex = self.transposition_dict[interval_name]["reindex"]
        new_x = reindex[x]
        if to_tensor:
            new_x = torch.tensor(new_x, device=device, dtype=tdtype)
        return new_x

    def introduce_transposition(self, interval):
        interval_name = interval.quality + str(interval.number)
        step = [re.sub(r"[\#\-]", "", p) for p in self.classes_]
        alter = [p.count("#") - p.count("-") for p in self.classes_]
        transposed_pitches = []
        for s, a in zip(step, alter):
            try:
                n = pt.utils.music.transpose_note(s, a, interval)
            except:
                n = ("X", 0)
            transposed_pitches.append(n)
        transposed_pitches = np.array(transposed_pitches, dtype=[("step", "U2"), ("alter", int)])
        idx = np.arange(len(self.classes_))
        reindex = np.zeros(len(self.classes_), dtype=int)
        accepted_pi2m = idx[transposed_pitches["step"] != "X"]
        transposed_pitches = transposed_pitches[accepted_pi2m]
        reindex[accepted_pi2m] = self.encode(transposed_pitches)
        self.transposition_dict[interval_name] = {"reindex": reindex, "accepted_indices": accepted_pi2m}


class KeySignatureEncoder(object):
    def __init__(self):
        self.KEY_SIGNATURES = list(range(-7, 8))
        self.encode_dim = len(self.KEY_SIGNATURES)
        self.classes_ = np.unique(self.KEY_SIGNATURES)
        self.interval_name_to_index = {
            "P1": 0,
            "m2": 7,
            "M2": 2,
            "m3": -3,
            "M3": 4,
            "P4": -1,
            "A4": 8,
            "d5": -6,
            "P5": 1,
            "m6": -4,
            "M6": 3,
            "m7": -2,
            "M7": 5,
        }

    def encode(self, note_array):
        # check if note_array is structured array or if it just an array of integers
        if isinstance(note_array, np.ndarray):
            if note_array.dtype.names is None:
                ks_array = note_array
            else:
                ks_array = note_array["ks_fifths"]
        elif isinstance(note_array, torch.Tensor):
            ks_array = note_array.detach().cpu().numpy()
        else:
            raise ValueError("Input has to be a numpy array or a torch tensor")
        return np.searchsorted(self.classes_, ks_array)

    def decode(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return self.classes_[x]

    def transpose(self, x, interval):
        """
        Transpose key signature by an interval.

        Parameters
        ----------
        x : numpy array or torch tensor
            Key signature integer labels.
        interval : partitura.Interval or str
            The interval by which to transpose the key signature.
        """
        to_tensor = False
        device = None
        if isinstance(x, torch.Tensor):
            device = x.device
            tdtype = x.dtype
            x = x.detach().cpu().numpy()
            to_tensor = True
        if not isinstance(interval, str):
            # Quality is any of the following: P, M, m, A, d
            quality = re.findall(r"[PMAmd]", interval)[0]
            number = int(re.findall(r"\d+", interval)[0])
            interval = Interval(number, quality)
            interval = interval.quality + str(interval.number)
        assert interval in self.interval_name_to_index.keys(), f"Interval {interval} not supported"
        new_x = self.decode(x) + self.interval_name_to_index[interval]
        assert np.all(np.isin(new_x, self.classes_)), "Key signature transposition is out of range."
        new_x = self.encode(new_x)
        if to_tensor:
            new_x = torch.tensor(new_x, device=device, dtype=tdtype)
        return new_x


class CadenceEncoder(object):
    """
    Encodes cadences to integer labels.

    The accepted cadences are:
    - PAC (Perfect Authentic Cadence)
    - IAC (Imperfect Authentic Cadence)
    - HC (Half Cadence)
    - DC (Deceptive Cadence)
    - EC (Evaded Cadence)
    - PC (Plagal Cadence)

    The encoding is:
    - No cadence: 0
    - PAC: 1
    - IAC: 2
    - HC: 3
    - DC/EC/PC: 4 (all grouped together because they are sparse in datasets)
    """
    def __init__(self):
        self.cadences = {
            "": 0,
            "PAC": 1,
            "IAC": 2,
            "HC": 3,
            "DC": 4,
            "EC": 4,
            "PC": 4,
        }
        self.accepted_cadences = np.array(["", "PAC", "IAC", "HC", "DC/EC/PC"])
        self.encode_dim = len(np.unique(list(self.cadences.values())))

    def encode(self, note_array, cadences):
        """
        Encodes a note array with cadences to integer labels.

        Parameters
        ----------
        note_array : numpy structured array
            A note array.
        cadences : list
            A list of partitura.Cadence objects.

        """
        labels = torch.zeros(len(note_array), dtype=torch.long)
        for cadence in cadences:
            labels[note_array["onset_div"] == cadence.start.t] = self.cadences[cadence.text]
        return labels

    def encode_from_text(self, text):
        return self.cadences[text]

    def decode(self, x):
        """
        Decodes integer labels to cadences.

        Parameters
        ----------
        x: numpy array or torch tensor
            Cadence Integer labels.

        Returns
        -------
        out: numpy array
            Cadence strings.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return self.accepted_cadences[x]


def transpose_note_array(note_array, interval):
    """
    Transpose a note array by a given interval.

    Parameters
    ----------
    note_array : numpy structured array
        A note array.
    interval : partitura.Interval
        The interval by which to transpose the note array.

    Returns
    -------
    out : numpy structured
    """
    out = note_array.copy()
    semitones = interval.semitones


    out["pitch"] = np.remainder(note_array["pitch"] + semitones, 127)
    if "step" in out.dtype.names:
        step = out["step"]
        alter = out["alter"]
        new_step, new_alter = np.vectorize(pt.utils.music.transpose_note)(step, alter, interval)
        out["step"] = new_step
        out["alter"] = new_alter
        out["octave"] = out["pitch"] // 12 - 1

    if "ks_fifths" in out.dtype.names:
        transpose_intervals = {
            (1, "P"): 0,
            (2, "m"): 7,
            (2, "M"): 2,
            (3, "m"): -3,
            (3, "M"): 4,
            (4, "P"): -1,
            (5, "d"): -6,
            (5, "P"): 1,
            (6, "m"): -4,
            (6, "M"): 3,
            (7, "m"): -2,
            (7, "M"): 5,
        }
        assert (interval.number, interval.quality) in transpose_intervals.keys(), f"Interval {interval} not supported for key signature transposition"
        out["ks_fifths"] = note_array["ks_fifths"] + transpose_intervals[(interval.number, interval.quality)]
        assert np.all(out["ks_fifths"] >= -7) and np.all(out["ks_fifths"] <= 7), "Key signature transposition out of range"
    return out
