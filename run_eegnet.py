import json
import wandb
import numpy as np
from pathlib import Path
from data.utils import *
from eegnet.torch_eegnet import *
from torch.utils.data import ConcatDataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

print("Initialize")
with open("eegnet/eegnet_conf.json",'r') as fconf:
	conf = json.load(fconf)

SAMPLING_RATE = {'bciciv':128, 'vepess':512}
conf['SAMPLING_RATE'] = SAMPLING_RATE[conf['DATA']]
NB_CLASSES = {'bciciv':4, 'vepess':2}
conf['NB_CLASSES'] = NB_CLASSES[conf['DATA']]
SIGNAL_LENGTH = {'bciciv':448, 'vepess':512}
conf['SIGNAL_LENGTH'] = SIGNAL_LENGTH[conf['DATA']]
NB_CHANS = {'bciciv':25, 'vepess':70}
conf['NB_CHANS'] = NB_CHANS[conf['DATA']]

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
wandb.define_metric("train accuracy", summary="mean")
wandb.define_metric("val accuracy", summary="mean")
model = EEGNet(config.SAMPLING_RATE, config.NB_CLASSES, config.SIGNAL_LENGTH, config.NB_CHANS)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), config.LEARNING_RATE)
wandb.watch(model, log="all")

print("Loading data")
gen_train_ds = GenDataset(config.DATA, config.GEN_RUN)
if config.DATA == 'vepess':
	src_train_ds = VepessDataset(config.N_SUBJECTS,True,partition='train')
	val_ds = VepessDataset(config.N_SUBJECTS,True,partition='val')
	test_ds = VepessDataset(config.N_SUBJECTS,True,partition='test')
else:
	src_train_ds = BCICIV2aDataset(config.N_SUBJECTS,True,partition='train')
	val_ds = BCICIV2aDataset(config.N_SUBJECTS,True,partition='val')
	test_ds = BCICIV2aDataset(config.N_SUBJECTS,True,partition='test')

val_dl = DataLoader(val_ds,batch_size=config.EVAL_BATCH_SIZE,shuffle=False)
test_dl = DataLoader(test_ds,batch_size=config.EVAL_BATCH_SIZE,shuffle=False)

print("Training")
if config.SETTING == 'DEFAULT':
	train_ds = src_train_ds
	train_dl = DataLoader(train_ds,batch_size=config.TRAIN_BATCH_SIZE,shuffle=True)
	run(model, device, train_dl, val_dl, optimizer, config, wandb)
elif config.SETTING == 'PRETRAIN':
	train_ds = gen_train_ds
	train_dl = DataLoader(train_ds,batch_size=config.TRAIN_BATCH_SIZE,shuffle=True)
	run(model, device, train_dl, val_dl, optimizer, config, wandb)
	train_ds = src_train_ds
	train_dl = DataLoader(train_ds,batch_size=config.TRAIN_BATCH_SIZE,shuffle=True)
	run(model, device, train_dl, val_dl, optimizer, config, wandb)
elif config.SETTING == 'DOUBLE':
	train_ds = ConcatDataset([src_train_ds, gen_train_ds])
	train_dl = DataLoader(train_ds,batch_size=config.TRAIN_BATCH_SIZE,shuffle=True)
	run(model, device, train_dl, val_dl, optimizer, config, wandb)
else:
	print("SETTING must be DEFAULT, PRETRAIN or DOUBLE.")
	exit(1)
