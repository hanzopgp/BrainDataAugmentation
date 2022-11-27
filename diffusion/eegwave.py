import torch
from torch.nn import Module, Conv2d, ModuleList, Sequential, Linear, SiLU, ReLU, Embedding
from torch.nn.functional import one_hot

def diffusion_step_embedding(lambd:int):
	"""
		Args:
			lambd: log SNR at time step t. Shape (B)
		Returns:
			Diffusion step embedding. Shape (B, 128)
	"""
	idx = 10**(torch.arange(64,device=lambd.device) * 4/63)
	tmp = idx * lambd.reshape(lambd.size(0),1)
	return torch.cat((torch.sin(tmp), torch.cos(tmp)),dim=-1)

class ResidualLayer(Module):
	def __init__(self,C:int,i:int,n:int,K:int):
		"""
			Args:
				C: residual channels
				i: layer index
				n: number of layers in a block
				K: dilated convolution filter size
		"""
		super().__init__()

		self.C = C
		self.i = i
		self.n = n

		self.lambd_linear = Linear(512,C)
		self.bidilconv = Conv2d(C,2*C,(1,K),padding="same",dilation=2**i%n)
		self.resconv = Conv2d(C,C,1)
		self.skipconv = Conv2d(C,C,1)

	def forward(self,x:torch.Tensor,lambd:torch.Tensor):
		"""
			Args:
				x: Input tensor. Shape (B, C, 1, L)
				lambd: log SNR at time step t. Shape (B, 512)
			Returns:
				Output tensor. Shape (B, C, 1, L)
		"""
		lambd = self.lambd_linear(lambd) # (B, C)
		lambd = lambd.reshape(*lambd.shape, 1, 1) # (B, C, 1, L)
		
		xt = x + lambd # (B, C, 1, L)
		xt = self.bidilconv(xt) # (B, 2C, 1, L)

		xt = xt[:,:self.C].tanh() * xt[:,self.C:].sigmoid()
		res = self.resconv(xt)
		skip = self.skipconv(xt)

		return res, skip

class EEGWave(Module):
	def __init__(self,n_class:int=2,n_subject:int=18,
			N:int=32,n:int=7,C:int=64,E:int=70,K:int=3):
		"""
			Args:
				n_class
				n_subject
				N: number of residual layers
				n: number of layers in a block
				C: residual channels
				E: input channels
				K: dilated convolution filter size
		"""
		super().__init__()

		self.n_class = n_class
		self.n_subject = n_subject
		self.N = N
		self.n = n
		self.C = C
		self.E = E
		self.K = K

		self.input_conv = Sequential(
			Conv2d(1,C,(E,1)),
			SiLU()
		)
		self.lambd_linear = Sequential(
			Linear(128,512),
			SiLU(),
			Linear(512,512),
			SiLU()
		)
		self.class_condition_emb = Embedding(n_class,512)
		self.subject_condition_emb = Embedding(n_subject,512)
		self.residual_layers = ModuleList([ResidualLayer(C,i,n,K) for i in range(N)])
		self.outconv = Sequential(
			Conv2d(C,C,1),
			SiLU(),
			Conv2d(C,E,1)
		)

	def forward(self,x:torch.Tensor,lambd:torch.Tensor,
			class_condition:torch.Tensor=None,
			subject_condition:torch.Tensor=None):
		"""
			Args:
				x: Input tensor. Shape (B, E, L)
				lambd: log SNR at time step t. Shape (B)
				class_condition: Shape (B)
				subject_condition: Shape (B)
			Returns:
				Output tensor. Shape (B, E, L)
		"""
		x = x.unsqueeze(1) # (B, 1, E, L)
		x = self.input_conv(x) # (B, C, 1, L)

		lambd = diffusion_step_embedding(lambd) # (B, 128)
		lambd = self.lambd_linear(lambd) # (B, 512)
		enc = lambd
		if class_condition is not None:
			cl = self.class_condition_emb(class_condition.long())
			enc = enc + cl
		if subject_condition is not None:
			sj = self.subject_condition_emb(subject_condition.long())
			enc = enc + sj

		all_skips = torch.zeros_like(x)
		for residual_layer in self.residual_layers:
			x, skip = residual_layer(x,enc) # (B, C, 1, L), (B, C, 1, L)
			all_skips = all_skips + skip

		output = self.outconv(all_skips) # (B, E, 1, L)
		output = output.squeeze(2) # (B, E, L)

		return output
