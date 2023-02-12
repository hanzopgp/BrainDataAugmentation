import torch
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
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

def _log1mexp(x):
	return torch.where(
			x > torch.log(torch.tensor(2,device=x.device)),
			torch.log1p(-torch.exp(-x)),
			torch.log(-torch.expm1(-x))
		)

class Diffusion(Module):
	def __init__(self, function_approximator, T=1024):
		"""
			Args:
				function_approximator: Neural network predicting x
				T: number of ancestral sampling steps
		"""
		super().__init__()
		self.function_approximator = function_approximator
		self.T = T

	def _schedule(self, t):
		"""
			Args:
				t: Diffusion time step. Shape (B)
			Returns:
				alpha: Shape (B)
				sigma: Shape (B)
				lambd: Shape (B)
				w: Shape (B)
			This scheduling function takes t between 0 and 1, and is therefore independent on the total number of sampling steps T
		"""
		s = 0.008
		_t = (t+s)/(1+s)
		alpha = torch.cos(0.5*np.pi*_t)
		alpha2 = torch.square(alpha)
		sigma2 = 1 - alpha2
		sigma = torch.sqrt(sigma2)
		lambd = (alpha2/sigma2).log()
		w = torch.maximum(torch.tensor(1,device=t.device),alpha2/sigma2)
		return alpha, sigma, lambd, w

	def compute_loss(self, x, cl=None, sj=None):
		t = torch.rand(x.size(0),device=x.device)
		epsilon = torch.normal(0,1,x.shape,device=x.device)
		alpha, sigma, lambd, w = self._schedule(t)
		with torch.no_grad():
			# noising
			_shape = (x.size(0),*(1,)*(len(x.shape)-1))
			z = alpha.view(_shape)*x + sigma.view(_shape)*epsilon
		# denoising
		pred = self.function_approximator(z, lambd, cl, sj)
		mse = torch.square(x - pred).mean((-2,-1))
		l = (w*mse).mean()
		return l, (z, torch.clip(pred,-1,1), mse.mean())

	def _step_sampling(self, s, t, zt, gamma, cl=None, sj=None, last_step=False):
		alpha_t, sigma_t, lambd_t, _ = self._schedule(t)
		alpha_s, sigma_s, lambd_s, _ = self._schedule(s)
		_shape = (zt.size(0),*(1,)*(len(zt.shape)-1))
		# compute distribution
		d_lambd = lambd_t - lambd_s
		e1 = torch.exp(d_lambd)
		e2 = -torch.expm1(d_lambd)
		x_hat = self.function_approximator(zt, lambd_t, cl, sj)
		mean = (e1*(alpha_s/alpha_t)).view(_shape)*zt + (e2*alpha_s).view(_shape)*x_hat
		log_e2 = _log1mexp(-d_lambd)
		log_sigma_st = log_e2 + 2*torch.log(sigma_s)
		log_sigma_ts = log_e2 + 2*torch.log(sigma_t)
		log_std = ((1-gamma)*log_sigma_st + gamma*log_sigma_ts)*0.5
		std = torch.exp(log_std)
		# sample
		epsilon = torch.normal(0,1,zt.shape,device=zt.device)
		zs = mean + std.view(_shape) * epsilon if not last_step else mean
		zs = torch.clip(zs,-1,1)
		return zs

	def ancestral_sampling(self, init_z, gamma=0.1, cl=None, sj=None):
		with torch.no_grad():
			z = init_z.clone()
			for t in torch.arange(1,0,-1/self.T):
				z = self._step_sampling(
					torch.ones(init_z.size(0),device=init_z.device) * (t-1/self.T),
					torch.ones(init_z.size(0),device=init_z.device) * t,
					z,gamma,cl,sj,
					t<=1/self.T
				)
		return z

	def forward(self, signal_length:int, gamma:float,
			class_conditioning:torch.tensor=None, subject_conditioning:torch.tensor=None):
		E = self.function_approximator.E
		init_z = torch.normal(0,1,(1,E,signal_length),device=next(self.parameters()).device)
		return self.ancestral_sampling(init_z, gamma=gamma, cl=class_conditioning, sj=subject_conditioning)

def train(model, device, dataloader, optimizer, wandb, class_conditioning, subject_conditioning):
	model.train()
	for i, (x,cl,sj) in tqdm(enumerate(dataloader),total=len(dataloader)):
		x = x.to(device,dtype=torch.float32)
		cl = cl.to(device,dtype=torch.long) if class_conditioning else None
		sj = sj.to(device,dtype=torch.long) if subject_conditioning else None
		l, (z, pred, mse) = model.compute_loss(x, cl, sj)
		wandb.log({
				"train diffusion loss": l.item(),
				"train mse loss": mse.item(),
				"x histogram": wandb.Histogram(x.detach().cpu().numpy()),
				"z histogram": wandb.Histogram(z.detach().cpu().numpy()),
				"pred histogram": wandb.Histogram(pred.detach().cpu().numpy())
			})
		optimizer.zero_grad()
		l.backward()
		clip_grad_norm_(model.parameters(), 1)
		optimizer.step()

def val(model, device, dataloader, wandb, class_conditioning, subject_conditioning):
	model.eval()
	with torch.no_grad():
		for i, (x,cl,sj) in tqdm(enumerate(dataloader),total=len(dataloader)):
			x = x.to(device,dtype=torch.float32)
			cl = cl.to(device,dtype=torch.long) if class_conditioning else None
			sj = sj.to(device,dtype=torch.long) if subject_conditioning else None
			l, (_, _, mse) = model.compute_loss(x, cl, sj)
			wandb.log({
					"val diffusion loss": l.item(),
					"val mse loss": mse.item(),
				})
