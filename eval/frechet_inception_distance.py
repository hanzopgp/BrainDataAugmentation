import torch
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_fid(
	model: torch.nn.Module,	# classification model (i.e. EEGNet)
	test_dl,
	gen_dl,
	device,
	nb_test,
	nb_gen,
	eps = 1e-6,
) -> float:
	""" Compute Frenchet Inception Distance between two batches of tensors `x1` and `x2`
		using `model` to output feature vectors for `x1` and `x2` from the
		second-to-last hidden layer (just before classification layer with softmax).
		Return the FID score for `x1` and `x2` """
	
	out1 = get_activations(model, device, test_dl, nb_test)
	out2 = get_activations(model, device, gen_dl, nb_gen)

	_s = min(out1.shape[0],out2.shape[0])
	out1 = out1[:_s]
	out2 = out2[:_s]

	m1, m2 = np.mean(out1, axis=0), np.mean(out2, axis=0)
	s1, s2 = np.cov(out1, rowvar=False), np.cov(out2, rowvar=False)

	diff = m1 - m2
	covmean, _ = sqrtm(np.dot(s1, s2), disp=False)

	if not np.isfinite(covmean).all():
		# Add small offset to counter infinite values in covmean matrix
		offset = np.eye(s1.shape[0]) * eps
		covmean = sqrtm(np.dot(s1 + offset, s2 + offset))

	if np.iscomplexobj(covmean):
		# If covmean has imaginary part due to numerical error,
		# retrieve only real part
		covmean = covmean.real

	return np.dot(diff, diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean)


def get_activations(model, device, dataloader, nb_samples):
	activation = {}
	def get_activation(name):
		def hook(model, input, output):
			activation[name] = input[0].detach()
		return hook
	model.eval()
	pred_arr = np.empty((nb_samples, model.clf.in_features))
	model.clf.register_forward_hook(get_activation('fid'))
	start_idx = 0
	with torch.no_grad():
		for i, (x,_,_) in tqdm(enumerate(dataloader)):
			x = x.to(device,dtype=torch.float32)
			_ = model(x)
			out = activation['fid']
			pred_arr[start_idx:start_idx+x.size(0)] = out.cpu().numpy()
			start_idx += x.size(0)
	return pred_arr