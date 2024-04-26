import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets

os.makedirs('output', exist_ok=True)
os.makedirs('models', exist_ok=True)

transform = transforms.Compose([
	transforms.Resize(64),
	transforms.ToTensor(),
	transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	])


class Gen(nn.Module):
	def __init__(self, latent_dim):
		super(Gen, self).__init__()
		self.l1 = nn.Linear(latent_dim, 128*4*4)
		self.conv_blocks = nn.Sequential(
			#128, 4, 4
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(0.1),
			#64, 8, 8
			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(0.1),
			#32, 16, 16

			nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(0.1),
			#16, 32, 32

			nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
			nn.Tanh()
			#3, 64, 64
			)

	def forward(self, v):
		x = self.l1(v)
		x = x.view(x.shape[0], 128, 4, 4)
		x = self.conv_blocks(x)

		return x


class Dis(nn.Module):
	def __init__(self):
		super(Dis, self).__init__()
		def down_block(inp_filters, out_filters):
			block = [nn.Conv2d(inp_filters, out_filters, kernel_size=3, stride=1, padding=1),
					 nn.MaxPool2d(2),
					 nn.LeakyReLU(0.1)]
			return block

		self.model = nn.Sequential(
			#3, 64, 64
			*down_block(3, 16),
			#16, 32, 32
			*down_block(16, 32),
			#32, 16, 16
			*down_block(32, 64),
			#64, 8, 8
			*down_block(64, 128),
			#128, 4, 4
			nn.Flatten(),
			nn.Linear(128*4*4, 64),
			nn.LeakyReLU(0.1),
			nn.Linear(64, 1),
			nn.Sigmoid()
			)

	def forward(self, x):
		x = self.model(x)
		return x

device = 'cuda'
latent_dim = 120

gen_model = Gen(latent_dim).to(device)
dis_model = Dis().to(device)


batch_size = 32
dataloader = DataLoader(datasets.CIFAR10("/data/CIFAR10", train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)

lr = 0.0001
optimizer_G = torch.optim.Adam(gen_model.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(dis_model.parameters(), lr=lr)

loss = nn.MSELoss()

num_epochs = 50
for epoch in range(num_epochs):
	for i, (imgs, _) in enumerate(tqdm(dataloader, desc=f'Epoch {(epoch+1)}/{num_epochs}')):
		valid = torch.ones((imgs.shape[0], 1), requires_grad=False, device=device)
		fake = torch.zeros((imgs.shape[0], 1), requires_grad=False, device=device)
		real_imgs = imgs.to(device)

		optimizer_G.zero_grad()
		v = torch.randn(imgs.shape[0], latent_dim, device=device)
		gen_imgs = gen_model(v)
		g_loss = loss(dis_model(gen_imgs), valid)
		g_loss.backward()
		optimizer_G.step()

		optimizer_D.zero_grad()
		real_loss = loss(dis_model(real_imgs), valid)
		fake_loss = loss(dis_model(gen_imgs.detach()), fake)
		d_loss = real_loss+fake_loss
		d_loss.backward()
		optimizer_D.step()

		if i%500 == 0:
			with torch.no_grad():
				torch.manual_seed(1)
				v = torch.randn(imgs.shape[0], latent_dim, device=device)
				gen_imgs = gen_model(v)
				save_image(gen_imgs.data/2+0.5, f'output/{epoch+1}_{i}.png')

	torch.save(gen_model.state_dict(), f'models/gen_{epoch}.pt')
	torch.save(dis_model.state_dict(), f'models/dis_{epoch}.pt')
