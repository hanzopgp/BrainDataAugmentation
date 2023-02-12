import torch
import numpy as np
from tqdm import tqdm

def compute_is_batch(
	model: torch.nn.Module,	# classfication model that can predict input data
	data: torch.Tensor,		# input data being classified and evaluated, size (batch, chan, dim)
	eps=1e-10,
) -> float:
	""" Compute the Inception Score on generated data `data` in batch using `classifier`'s
	output. A small `eps` is added to each tensor's value to avoid zero logarithm """

	output = model(data)	# classifier's prediction, size (batch, n_classes)
	output = torch.nn.functional.softmax(output, dim=-1)
	p_y = torch.unsqueeze(torch.mean(output, dim=0), 0)
	# KL divergence
	kl = output * (torch.log(output + eps) - torch.log(p_y + eps))
	# average of sum over classes
	avg_kl = torch.mean(torch.sum(kl, dim=1))
	# undo the log
	scores = torch.exp(avg_kl)
	return torch.mean(scores).item()

def compute_is(model, device, gen_dl):
	model.eval()
	scores = np.zeros(len(gen_dl))
	nb_samples = 0
	for i, (x,_,_) in tqdm(enumerate(gen_dl)):
		x = x.to(device,dtype=torch.float32)
		nb_samples += x.size(0)
		scores[i] = compute_is_batch(model,x,eps=1e-10) * x.size(0)
	return scores.sum() / nb_samples, scores