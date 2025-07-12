import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data import create_data_loader
from utils.weights import weights_init
from generator import Generator
from discriminator import Discriminator

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils as vutils

def train_gan(args):
	dataloader = create_data_loader(args)

	device = torch.device('cuda:0' 
		if (torch.cuda.is_available() and args.ngpu>0) 
		else 'cpu')
 
	netG = Generator(args).to(device)

	if (device.type == 'cuda' and args.ngpu>1):
		netG = nn.DataParallel(netG, list(range(args.ngpu)))

	if args.netG:
		netG.load_state_dict(torch.load(args.netG))
	
	else:
		netG.apply(weights_init)

	netD = Discriminator(args).to(device)

	if (device.type == 'cuda' and args.ngpu>1):
		netD = nn.DataParallel(netD, list(range(args.ngpu)))

	if args.netD:
		netD.load_state_dict(torch.load(args.netD))
	
	else:
		netD.apply(weights_init)


	criterion = nn.BCELoss()
	optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
	optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

	# For input of generator in testing
	fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

	# convention for training
	real_label = 1.0
	fake_label = 0.0

	# training data for later analysis
	img_list= []
	G_losses = []
	D_losses = []
	iters = 0

	num_epochs = 150

	print('Starting Training Loop....')
	# For each epoch
	for e in range(args.num_epochs):
		# for each batch in the dataloader
			for i, data in enumerate(dataloader, 0):
				netD.zero_grad()

				real_data = data[0].to(device)

				batch_size = real_data.size(0)
				labels = torch.full((batch_size,), real_label, device=device)

	
				real_outputD = netD(real_data).view(-1)

				errD_real = criterion(real_outputD, labels)
				errD_real.backward()
				D_x = real_outputD.mean().item()

				noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
				fake_data = netG(noise)
				labels.fill_(fake_label)
				fake_outputD = netD(fake_data.detach()).view(-1)

				errD_fake = criterion(fake_outputD, labels)

				errD_fake.backward()
				D_G_z1 = fake_outputD.mean().item()

			
				errD = errD_real + errD_fake
				optimizerD.step()

				netG.zero_grad()

				labels.fill_(real_label)

				fake_outputD = netD(fake_data).view(-1)

				errG = criterion(fake_outputD, labels)

				errG.backward()

				D_G_z2 = fake_outputD.mean().item()

	
				optimizerG.step()

			
				if i%500==0:
					print(f'[{e+1}/{args.num_epochs}][{i+1}/{len(dataloader)}]\
						\tLoss_D:{errD.item():.4f}\
						\tLoss_G:{errG.item():.4f}\
						\tD(x):{D_x:.4f}\
						\tD(G(z)):{D_G_z1:.4f}/{D_G_z2:.4f}')

		
				G_losses.append(errG.item())
				D_losses.append(errD.item())

			
				if ((iters % 500== 0) or 
					((e == args.num_epochs -1) and (i==len(dataloader)-1))):
					with torch.no_grad():
						fake = netG(fixed_noise).detach().cpu()
						img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
				iters +=1

			if e%args.save_every==0:
			
				torch.save(netG.state_dict(), args.outputG)
				torch.save(netD.state_dict(), args.outputD)
				print(f'Made a New Checkpoint for {e+1}')

	torch.save(netG.state_dict(), args.outputG)
	torch.save(netD.state_dict(), args.outputD)
	print(f'Saved Final model at {args.outputG} & {args.outputD}')
	
	return img_list, G_losses, D_losses







		
	


