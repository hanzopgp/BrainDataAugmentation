from diffusion.distillation import *
from diffusion.diffusion import *
from diffusion.eegwave import *
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import json
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_checkpoint(savepath, device):
	checkpoint = torch.load(savepath)
	epoch = checkpoint['epoch']
	config = checkpoint['config']
	function_approximator = EEGWave(
		checkpoint['n_class'],
		checkpoint['n_subject'],
		checkpoint['N'],
		checkpoint['n'],
		checkpoint['C'],
		checkpoint['E'],
		checkpoint['K']
	)
	model = Diffusion(function_approximator, checkpoint['T'])
	model.load_state_dict(checkpoint['model_state_dict'])
	model.to(device)
	return epoch, config, model

def sample():
	with open("diffusion/sampling_conf.json",'r') as fconf:
		config = json.load(fconf)

	cp_path = Path(f"{os.path.dirname(os.path.abspath(__file__))}/diffusion/checkpoints/{config['checkpoint']}")
	epoch, config2, model = load_checkpoint(cp_path, device)
	flag_class_conditioning = "_c" if config2["CLASS_CONDITIONING"] else ""
	flag_subject_conditioning = "_s" if config2["SUBJECT_CONDITIONING"] else ""
	with open (Path(f"{os.path.dirname(os.path.abspath(__file__))}/data/{config['data']}_stats.json"),'r') as fstat:
		stats = json.load(fstat)
	nb_orig_samples = 4837 if config['data']=="vepess" else 2592
	sample_path = Path(f"{os.path.dirname(os.path.abspath(__file__))}/sampled/{config['data']}/{config['checkpoint'][:-4]}{flag_class_conditioning}{flag_subject_conditioning}_{config['set']}")
	sample_path.mkdir(parents=True, exist_ok=True)

	index_start = max(0,len(os.listdir(sample_path))-1)
	for s in stats:
		print(f"Subject {s}/{len(stats)}")
		for c in stats[s]:
			nb_samples_of_class_subject = int(stats[s][c] * config['nb_samples'] / nb_orig_samples)
			class_condition = torch.tensor([int(c)],dtype=torch.long,device=device) if flag_class_conditioning else None
			subject_condition = torch.tensor([int(s)],dtype=torch.long,device=device) if flag_subject_conditioning else None
			for index in tqdm(range(index_start, index_start+nb_samples_of_class_subject)):
				x_hat = model(config['signal_length'], config['gamma'],
					class_conditioning=class_condition, subject_conditioning=subject_condition)
				torch.save((x_hat.detach().cpu(),class_condition,subject_condition), f"{sample_path}/tensor{index}.pt")
			index_start = index_start + nb_samples_of_class_subject

	config["nb_samples"] = len(os.listdir(sample_path))
	with open(f"{sample_path}/sampling_conf.json",'w') as f:
		json.dump({**config,**config2}, f)

	return sample_path

if __name__ == '__main__':
	sample()