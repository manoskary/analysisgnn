"""roman_decode.py

Utility to reconstruct a full Roman numeral string from decoded predictions
of the five multi-task learning outputs:

    degree1   – primary scale degree  (M21_DEGREES vocabulary,
                 e.g. '1', '#4', '-7', 'None')
    degree2   – secondary scale degree / tonicisation denominator
                 (same M21_DEGREES vocabulary; 'None' means no secondary function)
    inversion – integer 0–3  (0 = root position)
    quality   – chord quality string  (CHORD_QUALITIES vocabulary)
    localkey  – local key string  (values of EMPIRICAL_KEYS, e.g. 'C', 'g', 'F#')

The output is a Roman numeral string compatible with music21 conventions and
the COMMON_ROMAN_NUMERALS / SIMPLE_NUMERAL_VOCABULARY vocabularies used throughout
this codebase, e.g. 'V7', 'ii6', 'viio7/V', 'Ger65', 'bII6'.

Assumptions
-----------
* All five inputs have already been decoded from the network output indices back
  to their respective class labels (strings / int) via the corresponding
  OutputRepresentation.classList lookups.
* The function does *not* validate that the combination is musically meaningful;
  it purely performs the string reconstruction.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Internal look-up tables
# ---------------------------------------------------------------------------

# Map Arabic digit (as string) → uppercase Roman numeral
_ARABIC_TO_ROMAN: dict[str, str] = {
    "1": "I",
    "2": "II",
    "3": "III",
    "4": "IV",
    "5": "V",
    "6": "VI",
    "7": "VII",
}

# Maps quality label (from CHORD_QUALITIES) to a 3-tuple:
#   (case, quality_marker, chord_type)
#
#   case           : "upper" → numeral is capitalised  (major-ish)
#                    "lower" → numeral is lower-case   (minor-ish)
#   quality_marker : string inserted between the numeral and the inversion figure
#                    e.g. "o" for diminished ("viio7"), "ø" for half-diminished
#   chord_type     : "triad" | "seventh" | "special"
#                    controls which inversion-figure table is used
_QUALITY_MAP: dict[str, tuple[str, str, str]] = {
    # Triads
    "major triad":                         ("upper", "",  "triad"),
    "minor triad":                         ("lower", "",  "triad"),
    "diminished triad":                    ("lower", "o", "triad"),
    "augmented triad":                     ("upper", "+", "triad"),
    # Seventh chords
    "dominant seventh chord":              ("upper", "",  "seventh"),
    "minor seventh chord":                 ("lower", "",  "seventh"),
    # Major seventh uses plain "7" / "65" / … figures (same as dominant in RN notation)
    "major seventh chord":                 ("upper", "",  "seventh"),
    "incomplete dominant-seventh chord":   ("upper", "",  "seventh"),
    "diminished seventh chord":            ("lower", "o", "seventh"),
    "half-diminished seventh chord":       ("lower", "ø", "seventh"),
    # minor-augmented tetrachord (mM7): treat as minor seventh for case,
    # but use "+" marker to signal the augmented fifth in the label
    "minor-augmented tetrachord":          ("lower", "+", "seventh"),
    # Augmented-sixth chords: fixed labels handled separately
    "augmented sixth":                     ("upper", "",  "special"),
    "German augmented sixth chord":        ("upper", "",  "special"),
    "French augmented sixth chord":        ("upper", "",  "special"),
    "Italian augmented sixth chord":       ("upper", "",  "special"),
    # Fallback
    "None":                                ("upper", "",  "triad"),
}

# Inversion figures for triads and seventh chords (figured-bass style)
_INVERSION_FIGURES: dict[str, dict[int, str]] = {
    "triad": {
        0: "",
        1: "6",
        2: "64",
    },
    "seventh": {
        0: "7",
        1: "65",
        2: "43",
        3: "2",
    },
}

# Fixed labels for augmented-sixth chords (position encodes inversion already)
_AUG6_LABELS: dict[str, str] = {
    "Italian augmented sixth chord": "It6",
    "German augmented sixth chord":  "Ger65",
    "French augmented sixth chord":  "Fr43",
    "augmented sixth":               "It6",   # generic fallback → Italian
}

# Natural scale-degree quality by key mode for secondary-function denominator casing
# major key: degrees 1,4,5 → major (upper); 2,3,6,7 → minor/dim (lower)
# natural minor: degrees 3,6,7 → major (upper); 1,2,4,5 → minor/dim (lower)
_MAJOR_DEGREE_CASE: dict[str, str] = {
    "1": "upper", "2": "lower", "3": "lower",
    "4": "upper", "5": "upper", "6": "lower", "7": "lower",
}
_MINOR_DEGREE_CASE: dict[str, str] = {
    "1": "lower", "2": "lower", "3": "upper",
    "4": "lower", "5": "lower", "6": "upper", "7": "upper",
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_degree(degree_str: str) -> tuple[str, str]:
    """Parse a M21_DEGREES string into (rn_accidental_prefix, arabic_digit).

    Leading '-' characters are converted to 'b' (flat prefix in RN notation).
    Leading '#' characters are kept as-is.

    Returns ('', '') when *degree_str* is 'None' or empty.

    Examples
    --------
    >>> _parse_degree('5')
    ('', '5')
    >>> _parse_degree('#4')
    ('#', '4')
    >>> _parse_degree('-2')
    ('b', '2')
    >>> _parse_degree('--7')
    ('bb', '7')
    >>> _parse_degree('None')
    ('', '')
    """
    if not degree_str or degree_str == "None":
        return ("", "")

    acc = ""
    rest = degree_str
    while rest.startswith("-"):
        acc += "b"
        rest = rest[1:]
    while rest.startswith("#"):
        acc += "#"
        rest = rest[1:]

    return acc, rest


def _degree_to_roman(degree_str: str, case: str) -> str:
    """Convert a M21_DEGREES string to a Roman numeral with accidental prefix.

    Parameters
    ----------
    degree_str : str
        Degree as found in M21_DEGREES, e.g. '5', '#4', '-7', 'None'.
    case : str
        'upper' for a capitalised numeral, 'lower' for lower-case.

    Returns
    -------
    str
        Roman numeral string, e.g. 'V', '#IV', 'bvii'.
        Returns '' when *degree_str* is 'None'.
    """
    acc, digit = _parse_degree(degree_str)
    if not digit:
        return ""
    base = _ARABIC_TO_ROMAN.get(digit, "?")
    numeral = acc + (base if case == "upper" else base.lower())
    return numeral


def _secondary_case(digit: str, localkey: str) -> str:
    """Determine the appropriate RN case for a secondary-function denominator.

    Uses the natural scale-degree quality for the given key mode (major /
    minor), falling back to 'upper' for unknown inputs.
    """
    is_minor = bool(localkey) and localkey[0].islower()
    table = _MINOR_DEGREE_CASE if is_minor else _MAJOR_DEGREE_CASE
    return table.get(digit, "upper")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode_roman_numeral(
    degree1: str,
    degree2: str,
    inversion: int | str,
    quality: str,
    localkey: str,
    *,
    include_key: bool = False,
) -> str:
    """Reconstruct a Roman numeral string from five decoded task predictions.

    Parameters
    ----------
    degree1 : str
        Primary scale degree from the M21_DEGREES vocabulary,
        e.g. '1', '5', '#4', '-7'.  Use 'None' when unavailable.
    degree2 : str
        Secondary scale degree (tonicisation denominator) from the same
        vocabulary, e.g. '5' to produce '/V'.  Use 'None' for no secondary
        function.
    inversion : int or str
        Chord inversion: 0 = root position, 1 = 1st inversion,
        2 = 2nd inversion, 3 = 3rd inversion.
    quality : str
        Chord quality as in **CHORD_QUALITIES**, e.g. 'major triad',
        'dominant seventh chord', 'diminished seventh chord'.
    localkey : str
        Local key as in the values of **EMPIRICAL_KEYS**, e.g. 'C', 'g',
        'F#'.  Used only to determine the natural case of the secondary-
        function denominator (major key → uppercase, minor key → lowercase)
        and for the optional key prefix.
    include_key : bool, optional
        When *True* the result is prefixed with the local key followed by
        a colon, e.g. ``'C: V7'``.  Default is *False*.

    Returns
    -------
    str
        Roman numeral string compatible with music21 / AugmentedNet
        conventions, e.g. ``'V7'``, ``'ii6'``, ``'viio7/V'``, ``'Ger65'``,
        ``'bII6'``.  Returns ``''`` when *degree1* is ``'None'`` or when
        reconstruction is not possible.

    Examples
    --------
    >>> decode_roman_numeral('5', 'None', 0, 'dominant seventh chord', 'C')
    'V7'
    >>> decode_roman_numeral('7', 'None', 0, 'diminished seventh chord', 'C')
    'viio7'
    >>> decode_roman_numeral('2', 'None', 1, 'minor triad', 'C')
    'ii6'
    >>> decode_roman_numeral('5', '5', 0, 'dominant seventh chord', 'C')
    'V7/V'
    >>> decode_roman_numeral('5', '2', 0, 'dominant seventh chord', 'C')
    'V7/ii'
    >>> decode_roman_numeral('1', 'None', 0, 'major triad', 'C', include_key=True)
    'C: I'
    """

    # ------------------------------------------------------------------
    # Guard: missing primary degree
    # ------------------------------------------------------------------
    if not degree1 or degree1 == "None":
        return ""

    # Normalise inputs
    quality = quality or "None"
    inv_int = int(inversion) if inversion is not None else 0

    # ------------------------------------------------------------------
    # Retrieve quality properties
    # ------------------------------------------------------------------
    case, quality_marker, chord_type = _QUALITY_MAP.get(
        quality, ("upper", "", "triad")
    )

    # ------------------------------------------------------------------
    # Augmented-sixth chords: fixed labels, no degree processing needed
    # ------------------------------------------------------------------
    if chord_type == "special":
        rn = _AUG6_LABELS.get(quality, "It6")
        if include_key and localkey:
            rn = f"{localkey}: {rn}"
        return rn

    # ------------------------------------------------------------------
    # Build the primary Roman numeral from degree1 + case
    # ------------------------------------------------------------------
    numeral = _degree_to_roman(degree1, case)
    if not numeral:
        return ""

    # ------------------------------------------------------------------
    # Build the figure string:  quality_marker + inversion figure
    #
    # Examples:
    #   major triad, root pos    → "" + ""   = ""       → "I"
    #   minor triad, 1st inv     → "" + "6"  = "6"      → "ii6"
    #   dim triad,   root pos    → "o" + ""  = "o"      → "viio"
    #   dim 7th,     root pos    → "o" + "7" = "o7"     → "viio7"
    #   half-dim 7th, 1st inv    → "ø" + "65" = "ø65"   → "iiø65"
    #   dom 7th,     2nd inv     → "" + "43" = "43"     → "V43"
    # ------------------------------------------------------------------
    figures = _INVERSION_FIGURES.get(chord_type, _INVERSION_FIGURES["triad"])
    inv_fig = figures.get(inv_int, figures[0])
    figure_str = quality_marker + inv_fig

    rn = numeral + figure_str

    # ------------------------------------------------------------------
    # Secondary function (tonicisation): append /denominator
    #
    # degree2 is the scale degree of the tonicised key *within* localkey.
    # Its case follows the natural scale-degree quality in localkey.
    # ------------------------------------------------------------------
    if degree2 and degree2 != "None":
        _, sec_digit = _parse_degree(degree2)
        sec_case = _secondary_case(sec_digit, localkey)
        sec_numeral = _degree_to_roman(degree2, sec_case)
        if sec_numeral:
            rn = rn + "/" + sec_numeral

    # ------------------------------------------------------------------
    # Optional key prefix
    # ------------------------------------------------------------------
    if include_key and localkey:
        rn = f"{localkey}: {rn}"

    return rn


def decode_roman_numerals_batch(
    degree1_list:   list[str],
    degree2_list:   list[str],
    inversion_list: list[int | str],
    quality_list:   list[str],
    localkey_list:  list[str],
    *,
    include_key: bool = False,
) -> list[str]:
    """Vectorised wrapper around :func:`decode_roman_numeral`.

    All input lists must have the same length.

    Parameters
    ----------
    degree1_list, degree2_list, inversion_list, quality_list, localkey_list :
        Per-frame decoded predictions for each of the five tasks.
    include_key : bool, optional
        Forwarded to :func:`decode_roman_numeral`.

    Returns
    -------
    list[str]
        One Roman numeral string per input frame.

    Examples
    --------
    >>> decode_roman_numerals_batch(
    ...     ['5', '1'],
    ...     ['None', 'None'],
    ...     [0, 1],
    ...     ['dominant seventh chord', 'major triad'],
    ...     ['C', 'G'],
    ... )
    ['V7', 'I6']
    """
    return [
        decode_roman_numeral(d1, d2, inv, q, lk, include_key=include_key)
        for d1, d2, inv, q, lk in zip(
            degree1_list, degree2_list, inversion_list, quality_list, localkey_list
        )
    ]
