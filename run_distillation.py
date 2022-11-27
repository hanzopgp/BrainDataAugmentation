from diffusion.distillation import *
from diffusion.diffusion import *
from data.utils import *
from diffusion.eegwave import *
from torch.utils.data import random_split
import numpy as np
import wandb
import json
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
with open("diffusion/distillation_conf.json",'r') as fconf:
	conf = json.load(fconf)

wandb.init(project="amal_diffusion",config=conf)
random_seed = np.random.choice(9999)
config = wandb.config
config.SEED = random_seed
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
torch.backends.cudnn.deterministic = True

print("Loading data")
if config.DATA == 'VEPESS':
	ds = VepessDataset(config.N_SUBJECTS,True)
	SIGNAL_LENGTH = 512
else:
	ds = BCICIV2aDataset(config.N_SUBJECTS,True)
	SIGNAL_LENGTH = 448

teacher_path = Path(f"{os.path.dirname(os.path.abspath(__file__))}/diffusion/checkpoints/diffusion_{config.TEACHER}.pch")
_, _, model = load_checkpoint(teacher_path ,device)
optimizer = torch.optim.Adam(model.parameters(), config.LEARNING_RATE)
wandb.watch(model, log="all")
train_ds, val_ds, test_ds = random_split(ds, [
		int(len(ds)*config.SPLIT[0]),
		int(len(ds)*config.SPLIT[1]),
		len(ds) - int(len(ds)*config.SPLIT[0]) - int(len(ds)*config.SPLIT[1])
	])
train_dl = DataLoader(train_ds,batch_size=config.TRAIN_BATCH_SIZE,shuffle=True)
val_dl = DataLoader(val_ds,batch_size=config.EVAL_BATCH_SIZE,shuffle=False)
test_dl = DataLoader(test_ds,batch_size=config.EVAL_BATCH_SIZE,shuffle=False)

print("Distilling")
distill(model, device, train_dl, val_dl, optimizer, config, wandb)