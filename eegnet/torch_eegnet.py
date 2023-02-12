import torch
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ELU, AvgPool2d, Dropout, Flatten, Linear, CrossEntropyLoss

class EEGNet(Module):
	def __init__(self, sampling_rate: int, N: int, L: int, C: int,
			F1=8, D=2, F2=16, dropout_rate=0.5):
		"""
			Args:
				sampling_rate: Sampling rate of data
				N: nb classes
				L: signal length
				C: nb channels
				F1: nb temporal filters
				D: depth multiplier
				F2: nb pointwise filters
				dropout_rate
		"""
		super().__init__()
		self.sampling_rate = sampling_rate
		self.N = N
		self.L = L
		self.C = C
		self.F1 = F1
		self.D = D
		self.F2 = F2
		self.dropout_rate = dropout_rate

		self.block1 = Sequential(OrderedDict([
			('conv', Conv2d(1, F1, (1, sampling_rate//2), padding='same', bias=False)), # (B, F1, C, L)
			('bn1', BatchNorm2d(F1)),
			('dconv', Conv2d(F1, D*F1, (C, 1), bias=False, groups=F1)), # (B, D*F1, 1, L)
			('bn2', BatchNorm2d(D*F1)),
			('elu', ELU()),
			('avgpool', AvgPool2d(1, 4)), # (B, D*F1, 1, L//4)
			('dropout', Dropout(self.dropout_rate))
		]))
		self.block2 = Sequential(OrderedDict([
			('sconv_d', Conv2d(D*F1, D*F1, (1, sampling_rate//8), padding='same', bias=False, groups=D*F1)), # (B, D*F1, 1, L//4)
			('sconv_p', Conv2d(D*F1, F2, (1, 1), padding='same', bias=False)), # (B, F2, 1, L//4)
			('bn', BatchNorm2d(F2)),
			('elu', ELU()),
			('avgpool', AvgPool2d(1, 8)), # (B, F2, 1, L//32)
			('dropout', Dropout(self.dropout_rate)),
			('flatten', Flatten()) # (B, F2*L//32)
		]))
		self.clf = Linear(F2*L//32, N) # (B, N)

	def forward(self,x:torch.Tensor):
		"""
			Args:
				x: input tensor, shape (B, C, L)
			Returns:
				y: logits shape (B, N)
		"""
		x = x.unsqueeze(1) # (B, 1, C, L)
		x = self.block1(x) # (B, D*F1, 1, L//4)
		x = self.block2(x) # (B, F2*L//32)
		y = self.clf(x) # (B, N)
		return y

def _constraint_linear_max_norm(linear, max_norm=0.25):
	with torch.no_grad():
		norm = linear.weight.norm().clamp(min=max_norm/2)
		desired = norm.clamp(max=max_norm)
		linear.weight *= (desired / norm)

def _constraint_filter_max_norm(conv, max_norm=1):
	with torch.no_grad():
		norm = conv.weight.norm(dim=2,keepdim=True).clamp(min=max_norm/2)
		desired = norm.clamp(max=max_norm)
		conv.weight *= (desired / norm)

def train(model, device, dataloader, optimizer, wandb, class_names, criterion=CrossEntropyLoss()):
	model.train()
	targets = []
	preds = []
	for i, (x,cl,sj) in tqdm(enumerate(dataloader),total=len(dataloader)):
		x = x.to(device,dtype=torch.float32)
		target = cl.to(device)
		logit = model(x)
		loss = criterion(logit, target)
		pred = logit.argmax(axis=-1)
		accuracy = (pred==target).float().mean()
		targets += target.tolist()
		preds += pred.tolist()
		wandb.log({"train loss": loss.item(), "train accuracy": accuracy.item()})
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		_constraint_filter_max_norm(model.block1.dconv)
		_constraint_linear_max_norm(model.clf)
	wandb.log({"train conf mat":wandb.plot.confusion_matrix(
		preds=preds, y_true=targets, class_names=class_names)})

def val(model, device, dataloader, wandb, class_names, criterion=CrossEntropyLoss()):
	model.eval()
	targets = []
	preds = []
	for i, (x,cl,sj) in tqdm(enumerate(dataloader),total=len(dataloader)):
		x = x.to(device,dtype=torch.float32)
		target = cl.to(device)
		logit = model(x)
		loss = criterion(logit, target)
		pred = logit.argmax(axis=-1)
		accuracy = (pred==target).float().mean()
		targets += target.tolist()
		preds += pred.tolist()
		wandb.log({"val loss": loss.item(), "val accuracy": accuracy.item()})
	wandb.log({"val conf mat":wandb.plot.confusion_matrix(
		preds=preds, y_true=targets, class_names=class_names)})

def save_checkpoint(epoch, config, model, savepath):
	device = next(model.parameters()).device 
	model.to('cpu')
	torch.save({
		'epoch': epoch,
		'config': {k:v for k,v in config.items()},
		'sampling_rate': model.sampling_rate,
		'N': model.N,
		'L': model.L,
		'C': model.C,
		'F1': model.F1,
		'D': model.D,
		'F2': model.F2,
		'dropout_rate': model.dropout_rate,
		'model_state_dict': model.state_dict()
	}, savepath)
	model.to(device)

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

def run(model, device, train_dl, val_dl, optimizer, config, wandb):
	class_names = ["Left","Right","Feet","Tongue"] if config.DATA=="bciciv" else ["Non-target","Target"]
	for epoch in range(1,1+config.EPOCHS):
		train(model, device, train_dl, optimizer, wandb, class_names)
		val(model, device, val_dl, wandb, class_names)
	savepath = Path(f"eegnet/checkpoints/eegnet_{config.SEED}_{epoch}.pch")
	save_checkpoint(epoch, config, model, savepath)
