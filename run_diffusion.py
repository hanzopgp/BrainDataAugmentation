from diffusion.diffusion import *
from diffusion.eegwave import *
from data.utils import *
from torch.utils.data import random_split
import numpy as np
import wandb
import json
import os
from eval import *
from run_sampling import sample
from eegnet.torch_eegnet import EEGNet
from pathlib import Path
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

print("Initialize")
with open("diffusion/diffusion_conf.json",'r') as fconf:
	conf = json.load(fconf)

if "checkpoint" in conf:
	savepath = Path(conf["checkpoint"])
	epoch, cp_conf, model = load_checkpoint(savepath, device)
	wandb.init(project="amal_diffusion",entity="amal_2223",config=cp_conf)
else:
	wandb.init(project="amal_diffusion",entity="amal_2223",config=conf)

random_seed = np.random.choice(9999)
config = wandb.config
config.SEED = random_seed
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
torch.backends.cudnn.deterministic = True

print("Loading data")
if config.DATA == 'VEPESS':
	train_ds = VepessDataset(config.N_SUBJECTS,True,partition='train')
	val_ds = VepessDataset(config.N_SUBJECTS,True,partition='val')
	test_ds = VepessDataset(config.N_SUBJECTS,True,partition='test')
	model = Diffusion(EEGWave(n_class=2,n_subject=18,E=70))
	SIGNAL_LENGTH = 512
else:
	train_ds = BCICIV2aDataset(config.N_SUBJECTS,True,partition='train')
	val_ds = BCICIV2aDataset(config.N_SUBJECTS,True,partition='val')
	test_ds = BCICIV2aDataset(config.N_SUBJECTS,True,partition='test')
	model = Diffusion(EEGWave(n_class=4,n_subject=9,E=25))
	SIGNAL_LENGTH = 448

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), config.LEARNING_RATE)
wandb.watch(model, log="all")
train_dl = DataLoader(train_ds,batch_size=config.TRAIN_BATCH_SIZE,shuffle=True)
val_dl = DataLoader(val_ds,batch_size=config.EVAL_BATCH_SIZE,shuffle=False)
test_dl = DataLoader(test_ds,batch_size=config.EVAL_BATCH_SIZE,shuffle=False)

def load_checkpoint(savepath, device):
	checkpoint = torch.load(savepath)
	epoch = checkpoint['epoch']
	config = checkpoint['config']
	model = EEGNet(
		checkpoint['sampling_rate'],
		checkpoint['N'],
		checkpoint['L'],
		checkpoint['C'],
		checkpoint['F1'],
		checkpoint['D'],
		checkpoint['F2'],
		checkpoint['dropout_rate'],
	)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.to(device)
	return epoch, config, model

_, conf_eeg, eegnet = load_checkpoint("eegnet/checkpoints/eegnet_default.pch", device)
eegnet.to(device)

is_t, _ = compute_is(eegnet, device, test_dl)
fid_t = compute_fid(eegnet, test_dl, val_dl, device, len(test_ds), len(val_ds))

sampling_conf = {
        "checkpoint": None,
        "nb_samples": 36,
        "data": config.DATA.lower(),
        "set": 1,
        "signal_length": SIGNAL_LENGTH,
        "gamma": 0.1
}

def run(model, device, train_dl, val_dl, optimizer, config, wandb):
	for epoch in range(1,1+config.EPOCHS):
		train(model, device, train_dl, optimizer, wandb, config.CLASS_CONDITIONING, config.SUBJECT_CONDITIONING)
		val(model, device, val_dl, wandb, config.CLASS_CONDITIONING, config.SUBJECT_CONDITIONING)
		savepath = Path(f"diffusion/checkpoints/diffusion_{config.SEED}_{epoch}.pch")
		save_checkpoint(epoch, config, model, savepath)

		if epoch%16==0:
			sampling_conf["checkpoint"] = f"diffusion_{config.SEED}_{epoch}.pch"
			with open("diffusion/sampling_conf.json",'w') as f:
				json.dump(sampling_conf, f)
			sample_path = sample()

			gen_ds = GenDataset(config.DATA.lower(), os.path.basename(sample_path))
			gen_dl = DataLoader(gen_ds,batch_size=sampling_conf["nb_samples"],shuffle=False)

			is_g, _ = compute_is(eegnet, device, gen_dl)
			fid_g = compute_fid(eegnet, test_dl, gen_dl, device, len(test_ds), len(gen_ds))

			wandb.log({
				"is_t": is_t,
				"fid_t": fid_t,
				"is_g": is_g,
				"fid_g": fid_g
			})

print("Training")
run(model, device, train_dl, val_dl, optimizer, config, wandb)


