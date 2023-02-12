import torch
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

def save_checkpoint(epoch, config, model, savepath):
	device = next(model.parameters()).device 
	model.to('cpu')
	torch.save({
		'epoch': epoch,
		'config': {k:v for k,v in config.items()},
		'T': model.T,
		'n_class': model.function_approximator.n_class,
		'n_subject': model.function_approximator.n_subject,
		'N': model.function_approximator.N,
		'n': model.function_approximator.n,
		'C': model.function_approximator.C,
		'E': model.function_approximator.E,
		'K': model.function_approximator.K,
		'model_state_dict': model.state_dict()
	}, savepath)
	model.to(device)

class Distillation(Module):
	def __init__(self, base_diffusion):
		"""
			Args:
				base_diffusion: Standard diffusion model
		"""
		super().__init__()
		self.teacher = base_diffusion
		self.student = deepcopy(base_diffusion)
		self.student.T = int(self.teacher.T / 2)
		for param in base_diffusion.parameters():
			param.requires_grad = False

	def forward(self, signal_length:int):
		return self.student(signal_length)

	def compute_loss(self, x, cl=None, sj=None):
		t = torch.randint(1,1+self.student.T,size=(x.size(0),),device=x.device) / self.student.T
		epsilon = torch.normal(0,1,x.shape,device=x.device)
		alpha, sigma, lambd, w = self.student._schedule(t)
		with torch.no_grad():
			# noising
			_shape = (x.size(0),*(1,)*(len(x.shape)-1))
			z = alpha.view(_shape)*x + sigma.view(_shape)*epsilon
			# target denoising
			t1 = t - 0.5/self.student.T
			t2 = t - 1/self.student.T
			alpha1, sigma1, lambd1, w1 = self.teacher._schedule(t1)
			alpha2, sigma2, lambd2, w2 = self.teacher._schedule(t2)
			x_hat1 = self.teacher.function_approximator(z, lambd, cl, sj)
			z1 = alpha1.view(_shape)*x_hat1 + (sigma1/sigma).view(_shape)*(z - alpha.view(_shape)*x_hat1)
			x_hat2 = self.teacher.function_approximator(z1, lambd1, cl, sj)
			z2 = alpha2.view(_shape)*x_hat2 + (sigma2/sigma1).view(_shape)*(z1 - alpha1.view(_shape)*x_hat2)
			_sigma_ratio = sigma2/sigma
			x_tilde = (z2 - _sigma_ratio.view(_shape)*z)/(alpha2 - _sigma_ratio*alpha).view(_shape)
		# denoising
		pred = self.student.function_approximator(z, lambd, cl, sj)
		mse = torch.square(pred - x_tilde).mean((-2,-1))
		l = (w*mse).mean()
		return l, (torch.clip(x_tilde,-1,1), torch.clip(pred,-1,1), mse.mean())

def distill(model, device, train_dl, val_dl, optimizer, config, wandb):
	for iteration in range(1,1+config.NB_DISTILLS):
		distillation = Distillation(model)
		nb_epochs = 2*config.EPOCHS if config.NB_DISTILLS-iteration<2 else config.EPOCHS
		for epoch in range(1,1+nb_epochs):
			train(distillation, iteration, device, train_dl, optimizer, wandb, config.CLASS_CONDITIONING, config.SUBJECT_CONDITIONING)
			val(distillation, iteration, device, val_dl, wandb, config.CLASS_CONDITIONING, config.SUBJECT_CONDITIONING)
			savepath = Path(f"diffusion/checkpoints/distilled_{distillation.student.T}_{config.SEED}_{epoch}.pch")
			save_checkpoint(epoch, config, distillation.student, savepath)
		model = distillation.student
	return model

def train(distillation, iteration, device, dataloader, optimizer, wandb, class_conditioning, subject_conditioning):
	distillation.student.train()
	for i, (x,cl,sj) in tqdm(enumerate(dataloader),total=len(dataloader)):
		x = x.to(device,dtype=torch.float32)
		cl = cl.to(device,dtype=torch.long) if class_conditioning else None
		sj = sj.to(device,dtype=torch.long) if subject_conditioning else None
		l, (x_tilde, pred, mse) = distillation.compute_loss(x, cl, sj)
		wandb.log({
				f"train {iteration} loss": l.item(),
				f"train {iteration} mse loss": mse.item(),
				f"{iteration} x histogram": wandb.Histogram(x.detach().cpu().numpy()),
				f"{iteration} x_tilde histogram": wandb.Histogram(x_tilde.detach().cpu().numpy()),
				f"{iteration} pred histogram": wandb.Histogram(pred.detach().cpu().numpy())
			})
		optimizer.zero_grad()
		l.backward()
		clip_grad_norm_(distillation.student.parameters(), 1)
		optimizer.step()

def val(distillation, iteration, device, dataloader, wandb, class_conditioning, subject_conditioning):
	distillation.student.eval()
	with torch.no_grad():
		for i, (x,cl,sj) in tqdm(enumerate(dataloader),total=len(dataloader)):
			x = x.to(device,dtype=torch.float32)
			cl = cl.to(device,dtype=torch.long) if class_conditioning else None
			sj = sj.to(device,dtype=torch.long) if subject_conditioning else None
			l, (_, _, mse) = distillation.compute_loss(x, cl, sj)
			wandb.log({
					f"val {iteration} loss": l.item(),
					f"val {iteration}  mse loss": mse.item(),
				})
