from diffusion.distillation import *
from diffusion.diffusion import *
from diffusion.eegwave import *
from pathlib import Path
import numpy as np
import json
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("diffusion/sampling_conf.json",'r') as fconf:
	config = json.load(fconf)

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

cp_path = Path(f"{os.path.dirname(os.path.abspath(__file__))}/diffusion/checkpoints/{config['checkpoint']}")
epoch, config2, model = load_checkpoint(cp_path, device)
sample_path = Path(f"{os.path.dirname(os.path.abspath(__file__))}/sampled/{config['data']}/{config['checkpoint'][:-4]}_{config['set']}")
sample_path.mkdir(parents=True, exist_ok=True)

with open(f"{sample_path}/sampling_conf.json",'w') as f:
	json.dump({**config,**config2}, f)

index_start = max(0,len(os.listdir(sample_path))-1)
for index in tqdm(range(index_start,index_start+config['nb_samples'])):
	x_hat = model(config['signal_length'], config['gamma'])
	torch.save(x_hat.detach().cpu(), f"{sample_path}/tensor{index}.pt")