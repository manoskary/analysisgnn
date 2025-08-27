import numpy as np
from itertools import groupby
from analysisgnn.utils import chord_to_intervalVector


int_vec_cadence_dict = { 
    "V/I maj" : [1, 2, 2, 2, 3, 0],
    "V/I min": [2, 1, 2, 3, 2, 0],
    "V7/I maj" : [2, 3, 3, 2, 4, 1],
    "V7/I min" : [2, 3, 3, 3, 3, 1],
    "V9/I min" : [3, 3, 5, 4, 4, 2],
    "IV/I maj" : [1, 2, 2, 2, 3, 0],
    "IV/I min" : [1, 2, 2, 2, 3, 0],
    "IV/I picard" : [2, 1, 2, 3, 2, 0],
    "IV/I dorian" : [0, 3, 2, 2, 2, 1],
    "V/VI" : [2, 3, 3, 3, 3, 1],
}



def get_cadences(note_array, step, window_size):
    '''
    A definition for match score analysis 
    
    Parameters
    ----------
    note_array : array(tuples)
        The note_array from the partitura analysis
    step : float
        The step for the analysis window.
    window_size : int
        The window size to draw samples from, usually a bar measure size.
    
    Returns
    -------
    The bar number where pivot cadences are found.
        
    '''
    # standard forward lim
    durations = [n['duration_beat'] for n in note_array if n['duration_beat']!=0]
    min_duration = min(durations)
    max_duration = max(durations)
    max_polyphony = max([len(list(item[1])) for item in groupby(note_array, key=lambda x: x[0])])
    forward_step_lim = int(max_duration / min_duration + max_polyphony)
    duration = note_array[-1]['onset_beat'] + max_duration - step
    
    # normalize duration
    duration = duration / step
    step_unit = 1
    dim = int(round((duration - (window_size * step_unit)) / step_unit) + 1)
    # 5 is the attributes number
    X = np.zeros((dim - 1, 6))
    Y = np.zeros(dim - 1)
    index = 0
    for window in range(1, dim - 1, step_unit):
        fix_start = window * step
        ind_list = list()
        note_list = list()
        if len(note_array[index:]) > forward_step_lim*window_size:
            look_in = forward_step_lim * window_size + index
        else:
            look_in = len(note_array)
        for ind, note in enumerate(note_array[index:look_in]):
            note_start = note['onset_beat'] #onset
            note_end = note['onset_beat'] + note['duration_beat'] #onset + duration
            fix_end = fix_start + (window_size * step)
            # check if note is in window
            if (fix_start <= note_start <= fix_end) or (note_start <= fix_start and note_end >= fix_start):
                    ind_list.append(ind)
                    note_list.append(note_array[index+ind]["pitch"])
        if ind_list != []:
            # compute interval Vector
            X[window] = np.array(chord_to_intervalVector(note_list))
            note_list = sorted(list(set(note_list)))
            bass_notes = list(map(lambda x: x%12, note_list[:2])) if len(note_list) >= 2 else None
            bass_int = abs(bass_notes[0]-bass_notes[1]) if bass_notes != None else 0
            Y[window] = True if bass_int == 5 or bass_int == 7 else False
            # update index
            index += min(ind_list)
    
    pivot_points = [ind+window_size/step for ind, value in enumerate(list(X))
                if list(value) in int_vec_cadence_dict.values() and Y[ind]]
    # have to automate x/6 to bar
    return pivot_points





if __name__ == "__main__":    
    import matplotlib.pyplot as plt
    import partitura
    import os
    dirname = os.path.dirname(__file__)
    par = lambda x: os.path.abspath(os.path.join(x, os.pardir))
    my_musicxml_file = os.path.join(par(par(dirname)), "samples", "xml", "k080-02.musicxml")
    step = 1
    bar = 4
    k = bar/step
    
    note_array = partitura.utils.ensure_notearray(partitura.load_musicxml(my_musicxml_file))
    cad = list(set(map(lambda x: int(x/k)+1, get_cadences(note_array, 1, 4))))
    print(cad)