import torch
#import mne
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
#import biosig
import json
from pathlib import Path

def load_vepess(s:int, freq=512):
	labelfile = f"{os.path.dirname(__file__)}/VEPESS/session/{s}/event_VEP_visual_oddball_session_{s}_task_Oddball_task_subjectLabId_{s:02d}_recording_1.tsv"
	datafile = f"{os.path.dirname(__file__)}/VEPESS/session/{s}/eeg_VEP_visual_oddball_session_{s}_task_Oddball_task_subjectLabId_{s:02d}_B_{s:02d}_VEP_recording_1.set"
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

_BCICIV_TYPES = {
	"0x0301": 1,
	"0x0302": 2,
	"0x0303": 3,
	"0x0304": 4
}

def load_bciciv2a(s:int, freq=128):
	datafile = f"{os.path.dirname(__file__)}/BCICIV_2a_gdf/a{s:02d}T.gdf"
	header = json.loads(biosig.header(datafile))
	raw = mne.io.read_raw_gdf(datafile)
	signal = raw.load_data().filter(4,38).resample(freq).get_data()

	labels = [0] * int(header['NumberOfSamples'])
	for event in header['EVENT']:
		if event['TYP'] in _BCICIV_TYPES:
			idx = int(event['POS'] * freq)
			labels[idx] = _BCICIV_TYPES[event['TYP']]
	labels = np.array(labels)
	results = []
	for idx in np.argwhere(labels!=0).flatten():
		x = signal[:,idx+freq//2:idx+freq*4]
		mu = x.mean(-1,keepdims=True)
		x = x - mu
		xmax = x.max(-1,keepdims=True)
		xmin = x.min(-1,keepdims=True)
		x = 2 * (x - xmin) / (xmax - xmin) - 1
		y = labels[idx]
		results.append((x,y,s))
	return results

class EEGDataset(Dataset):
	def __init__(self, n:int, cached=False):
		self.data = []
		self.cached = cached
		self.subject_labels = {(i+1):i for i in range(n)}

	def __len__(self):
		if self.cached:
			return len(os.listdir(self.cachedir))
		return len(self.data)

	def __getitem__(self,index):
		if self.cached:
			return self._parse_data(torch.load(f"{self.cachedir}/tensor{index}.pt"))
		return self._parse_data(self.data[index])

	def _parse_data(self,data):
		x, y, s = data
		return x, self.class_labels[y], self.subject_labels[s]

	def cache(self):
		for index, tensor in enumerate(self.data):
			torch.save(tensor, f"{self.cachedir}/tensor{index}.pt")

class VepessDataset(EEGDataset):
	def __init__(self, n:int, cached=False):
		super().__init__(n, cached)
		self.cachedir = Path(f"{os.path.dirname(__file__)}/pt_vepess")
		if not cached:
			for i in range(1,n+1):
				self.data += load_vepess(i)
		self.class_labels = {34:0, 35:1}

class BCICIV2aDataset(EEGDataset):
	def __init__(self, n:int, cached=False):
		super().__init__(n, cached)
		self.cachedir = Path(f"{os.path.dirname(__file__)}/pt_bciciv")
		if not cached:
			for i in range(1,n+1):
				self.data += load_bciciv2a(i)
		self.class_labels = {1:0, 2:1, 3:2, 4:3}
