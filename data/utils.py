import torch
import mne
import numpy as np

VEP_PATH = "VEPESS/session"

def load_vep(s:int, freq=512):
	labelfile = f"{VEP_PATH}/{s}/event_VEP_visual_oddball_session_{s}_task_Oddball_task_subjectLabId_{s:02d}_recording_1.tsv"
	datafile = f"{VEP_PATH}/{s}/eeg_VEP_visual_oddball_session_{s}_task_Oddball_task_subjectLabId_{s:02d}_B_{s:02d}_VEP_recording_1.set"
	raw = mne.io.read_raw_eeglab(datafile)
	signal = raw.filter(1,40).get_data()
	labels = [0] * int(raw.n_times)
	with open(labelfile,'r') as f:
		for line in f.readlines():
			label, timestamp, _ = line.split('\t',2)
			idx = int(float(timestamp) * freq)
			labels[idx] = int(label) if label in ['34','35'] else 0
	labels = np.array(labels)
	results = []
	for idx in np.argwhere(labels!=0).flatten():
		x = signal[:,idx:idx+freq]
		mu = x.mean(-1,keepdims=True)
		x = x - mu
		xmax = x.max(-1,keepdims=True)
		xmin = x.min(-1,keepdims=True)
		x = 2 * (x - xmin) / (xmax - xmin) - 1
		y = labels[idx]
		results.append((x,y,s))
	return results

