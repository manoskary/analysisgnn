import numpy as np
from analysisgnn.utils import chord_to_intervalVector


INTVEC_DICT = { 
    "V/I maj" : [1, 2, 2, 2, 3, 0],
    "V/I min": [2, 1, 2, 3, 2, 0],
    "V7/I maj" : [2, 3, 3, 2, 4, 1],
    "V7/I min" : [2, 3, 3, 3, 3, 1],
    "V9/I min" : [3, 3, 5, 4, 4, 2]
}

INTVEC_DOM = { 
    "V" : [0, 1, 1, 1, 0, 0],
    "V7": [0, 1, 2, 1, 1, 1],
    "V9" : [1, 1, 4, 1, 1, 2]
}



def h_cad_onset(tnotes, bar_in_beats, index):
	onset = index + bar_in_beats
	ind = np.where((tnotes["onset_beat"] <= onset ) & (tnotes["onset_beat"] + tnotes["duration_beat"] > onset))[0]
	if ind.size > 1:
		cp = np.argsort(tnotes[ind]["pitch"])[::-1]
		if chord_to_intervalVector(cp) in INTVEC_DOM.values():
			rp = [p for p in cp[:-1] if (p - cp[-1])%12 in [4, 7]]
		else:
			return None
		if len(rp)==2:
			for pp in rp:
				non = cp[np.where(cp["pitch"] == pp)]["onset"]	
				pitches = tnotes[np.where(np.isclose(tnotes["onset_beat"]+tnotes["duration_beat"], non) == True)]["pitch"]
				if 1 in list(map(lambda x : x - pp, pitches)):
					return onset
	return None


def p_cad_bass(tnotes, bar_in_beats, index):
	if index%bar_in_beats != 0:
		onset = index - index%bar_in_beats + bar_in_beats
	else:
		return None
	ind = np.where(tnotes["onset_beat"] == onset)[0]
	if ind.size > 0:
		bp = np.min(tnotes[ind]["pitch"])
		pnotes = tnotes[np.where(np.isclose(tnotes["onset_beat"]+tnotes["duration_beat"], onset) == True)]
		if pnotes.size > 0:
			# Search for V - I in first onset of measure.
			low = np.min(pnotes["pitch"])
			if low - bp in [7, -5]:	
				# Now we search for possible voice leading patterns.
				preonset = pnotes[np.argmin(pnotes["pitch"])]["onset_beat"]
				vl = tnotes[np.where((tnotes["onset_beat"] >= preonset) & (tnotes["onset_beat"] < onset))]
				vl = vl[np.where(np.isclose(vl["onset_beat"], np.max(vl["onset_beat"])) == True)]
				if (np.max(vl["pitch"]) - low)%12 in [4, 7, 10]:					
					if vl.size > 1:
						i = np.argmax(vl["pitch"])
						rr = vl["onset_beat"][i] + vl["duration_beat"][i]
					elif vl.size == 1:
						rr = vl["onset_beat"] + vl["duration_beat"]
					else:
						return None
					if tnotes[np.where(np.isclose(tnotes["onset_beat"], rr) == True)].size > 0:
						if np.max(vl["pitch"]) - np.max(tnotes[np.where(np.isclose(tnotes["onset_beat"], rr) == True)]["pitch"]) in [1, 2, -1, 0]:
							return onset
	return None	


def vl3cons(tnotes, bar_in_beats, index):
	onset = index + bar_in_beats
	VOICE_LEADING_INT = [[-3, 1], [3, -2], [-2, 1]]
	ind = np.where(tnotes["onset_beat"] == onset)[0]
	if ind.size > 1:
		cmin = np.min(tnotes[ind]["pitch"])
		cmax = np.max(tnotes[ind]["pitch"])
		pnotes = tnotes[np.where(np.isclose(tnotes["onset_beat"]+tnotes["duration_beat"], onset) == True)]
		# pnotes = tnotes[np.where(tnotes["onset_beat"]+tnotes["duration_beat"] == onset)]
		if pnotes.size > 0:
			pmax = np.max(pnotes["pitch"])
			ponset = pnotes[np.where(pnotes["pitch"] == pmax)]["onset_beat"][0]
		else:
			return None
		ppnotes = tnotes[np.where(np.isclose(tnotes["onset_beat"]+tnotes["duration_beat"], ponset) == True)]
		if ppnotes.size > 0:
			ppmax = np.max(ppnotes["pitch"])
			dc = [pmax - ppmax, cmax - pmax]		
			if dc in VOICE_LEADING_INT and cmin%12==cmax%12:
				return onset
	return None



def p_cad_delay(tnotes, bar_in_beats, index):
	onset = index + bar_in_beats
	ind = np.where(tnotes["onset_beat"] == onset)[0]
	if ind.size > 1:
		cp = np.argsort(tnotes[ind]["pitch"])[::-1]
		# Search for particular voicing
		if cp[0] - cp[1] in [3, 4, 15, 16] and cp[1]%12 == cp[-1]%12:
			pnotes = tnotes[np.where(np.isclose(tnotes["onset_beat"]+tnotes["duration_beat"], onset) == True)]
			if pnotes.size > 1:
				pp = np.argsort(pnotes["pitch"])[::-1]
				# Search for Voice Leading schema
				if pp[0] - cp[0] in [1, 2] and pp[1] - cp[1] == 2:
					minonset = np.min(pnotes["onset"])
					# Search for 
					if (cp[1] - np.min(tnotes[np.where(tnotes["onset_beat"]+tnotes["duration_beat"] == minonset)]["pitch"]))%12 in [7, 5]:
						return onset
	return None


def cad_delay(tnotes, bar_in_beats, index):
	onset = index + bar_in_beats
	ind = np.where(tnotes["onset_beat"] == onset)[0]
	if ind.size > 0:
		max_pitch = np.max(tnotes[ind]["pitch"])  
	else: 
		return None
	bnotes = tnotes[np.where((tnotes["onset_beat"]+tnotes["duration_beat"] > onset) & (tnotes["onset_beat"] < onset))] 
	if bnotes.size != 0 :
		low_note = bnotes[np.argmin(bnotes["pitch"])]
		min_pitch = low_note["pitch"]
		if max_pitch%12 - min_pitch%12 == 0:
			hp = tnotes[np.where(tnotes["onset_beat"]+tnotes["duration_beat"] == onset)]
			lp = tnotes[np.where(tnotes["onset_beat"]+tnotes["duration_beat"] == low_note["onset_beat"])]
			if hp.size > 0 and lp.size > 0:
				hp = np.max(hp["pitch"])
				lp = np.min(lp["pitch"])
				if hp - max_pitch in [2, -1] and lp - min_pitch in [7, -5]:
					return onset
	return None


def cad_onset(tnotes, bar_in_beats, index):
	onset = index + bar_in_beats - 1
	ind = np.where(tnotes["onset_beat"] == onset)[0]
	if ind.size > 1:
		last_onset_notes = tnotes[ind]
		max_pitch = np.max(last_onset_notes["pitch"])
		min_pitch = np.min(last_onset_notes["pitch"])
		if max_pitch%12 - min_pitch%12 == 0:
			p = tnotes[np.where(np.isclose(tnotes["onset_beat"]+tnotes["duration_beat"], onset) == True)]
			if p.size > 0:
				hp = np.max(p["pitch"])
				lp = np.min(p["pitch"])
				if hp - max_pitch == 2 and lp - min_pitch in [7, -5] and p[np.where(p["pitch"] == lp)]["onset_beat"][0]%1==0:					
					return onset
				elif hp - max_pitch in [7, -5] and p[np.where(p["pitch"] == hp)]["onset_beat"][0]%1==0:
					temp1 = np.intersect1d(tnotes[ np.where( tnotes["onset_beat"] <= onset )], tnotes[ np.where( tnotes["onset_beat"] >= onset - 1 )]) 
					temp2 = np.intersect1d(tnotes[ np.where( tnotes["onset_beat"] < onset - 1)], tnotes[ np.where( tnotes["onset_beat"]+tnotes["duration_beat"] >= onset-1 )])
					tnotes = np.union1d(temp1, temp2)
					if chord_to_intervalVector in INTVEC_DICT.values():						
						return onset
				elif vl4cons(tnotes, p, hp, max_pitch):					
					return onset
				else:
					pass
	return None


def loop_cons_notes(tnotes, p, hp, max_pitch, rel_int):
	if hp - max_pitch == rel_int:
		onset = p[np.where(p["pitch"] == hp)]["onset_beat"]
		p = tnotes[np.where(tnotes["onset_beat"]+tnotes["duration_beat"] == onset)]
		if p.size > 0:
			max_pitch = hp
			hp = np.max(p["pitch"])
			return p, hp, max_pitch
	else :
		return None

def vl4cons(tnotes, p, hp, max_pitch, rel = [-1, 1, 2]):
	for rel_int in rel:
		x = loop_cons_notes(tnotes, p, hp, max_pitch, rel_int)
		if x != None:
			p, hp, max_pitch = x
		else:
			return False
	return True			





def get_voice_leading(note_array, bar_in_beats):
	'''
	Does the intervalic analysis of a piece.
	
	Parameters:
	-----------
	note_array : Structured array
		A partitura note array from one or more parts
	bar_in_beats : float
		The bar duration in beats
	
	Returns:
	--------
	vl_beat_pos : list
		Onsets in Beats of Possible Voice Leading Occurences
	'''    
	# standard forward lim
	max_onset = np.max(note_array["onset_beat"])
	length = max_onset + np.max(note_array[np.where(note_array["onset_beat"] == max_onset)[0]]["duration_beat"]) - bar_in_beats
	step = 1 
	window_size = bar_in_beats + 1
	vl_beat_pos = list()
	for index in range(0, int(length), step): 
		temp1 = np.intersect1d(note_array[ np.where( note_array["onset_beat"] < index + window_size )], note_array[ np.where( note_array["onset_beat"] >= index )]) 
		temp2 = np.intersect1d(note_array[ np.where( note_array["onset_beat"] < index)], note_array[ np.where( note_array["onset_beat"]+note_array["duration_beat"] >= index )])
		tnotes = np.union1d(temp1, temp2)
		vl_beat_pos += filter(None, [
			p_cad_bass(tnotes, bar_in_beats, index),
			cad_delay(tnotes, bar_in_beats, index), 
			p_cad_delay(tnotes, bar_in_beats, index), 
			cad_onset(tnotes, bar_in_beats, index),
			h_cad_onset(tnotes, bar_in_beats, index),
			vl3cons(tnotes, bar_in_beats, index)
			])
	return sorted(list(set(vl_beat_pos)))



if __name__ == "__main__":
	import partitura
	import os

	dirname = os.path.dirname(__file__)
	par = lambda x: os.path.abspath(os.path.join(x, os.pardir))
	my_musicxml_file = os.path.join(par(par(dirname)), "samples", "xml", "mozart_piano_sonatas", "K284-2.musicxml")
	
	note_array = partitura.utils.ensure_notearray(partitura.load_musicxml(my_musicxml_file))
	bar_in_beats = 3
	print(list(map(lambda x: (int(x / bar_in_beats) + 1, x%bar_in_beats), get_voice_leading(note_array, bar_in_beats))))