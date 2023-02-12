from tqdm import tqdm
from eval import *
from data.utils import *
from eegnet.torch_eegnet import *
import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("eval/conf.json",'r') as fconf:
	conf = json.load(fconf)

_, config, model = load_checkpoint(conf['CHECKPOINT'], device)
model.to(device)

gen_ds = GenDataset(conf['DATA'], conf['GEN_RUN'])
if conf['DATA'] == 'vepess':
	val_ds = VepessDataset(conf['N_SUBJECTS'],True,partition='val')
	test_ds = VepessDataset(conf['N_SUBJECTS'],True,partition='test')
else:
	val_ds = BCICIV2aDataset(conf['N_SUBJECTS'],True,partition='val')
	test_ds = BCICIV2aDataset(conf['N_SUBJECTS'],True,partition='test')

gen_dl = DataLoader(gen_ds,batch_size=conf['BATCH_SIZE'],shuffle=False)
val_dl = DataLoader(val_ds,batch_size=conf['BATCH_SIZE'],shuffle=False)
test_dl = DataLoader(test_ds,batch_size=conf['BATCH_SIZE'],shuffle=False)

s, ss = compute_is(model, device, test_dl)
print(f"Inception score - test: {s}")

s, ss = compute_is(model, device, gen_dl)
print(f"Inception score - {conf['GEN_RUN']}: {s}")

fid = compute_fid(model, test_dl, val_dl, device, len(test_ds), len(val_ds))
print(f"FID score - val vs. test: {fid}")

fid = compute_fid(model, test_dl, gen_dl, device, len(test_ds), len(gen_ds))
print(f"FID score - {conf['GEN_RUN']}: {fid}")