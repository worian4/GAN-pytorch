import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import moviepy.editor as mp
from torchvision.utils import save_image
import time

from func import transform, weights_init
import models



manualSeed = random.randint(1, 10000)
print("\nRandom Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)

dataroot = 'colored'
dataname = 'results.txt'
sym = '\t'
workers = 12
image_size = 64
batch_size = 128
nz = 100
num_epochs = 500
lr = 0.0002
beta1 = 0.5
ngpu = 1
device = torch.device('cuda:0')
 
num_im = 64

n_extra_layers_g = 1
n_extra_layers_d = 0

dataset = dset.ImageFolder(root='./'+dataroot+'/', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

real_batch = next(iter(dataloader))

netG = models._netG_1(ngpu, nz, 3, image_size, n_extra_layers_g).to(device)
netD = models._netD_1(ngpu, nz, 3, image_size, n_extra_layers_d).to(device)

netG.apply(weights_init)
netD.apply(weights_init)


criterion = nn.BCELoss()

fixed_noise = torch.randn(num_im, nz, 1, 1, device=device)
fnoise = torch.randn(1, nz, 1, 1, device=device)

real_label = 1.
fake_label = 0.

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
loss_time = []
epochs = []
loader = []
perepoch = []

saved_iter = 0
iters = 0
time_per_iter = 0
time_per_epoch =0
tle = 0

low_Gloss = 100.0
low_Dloss = 100.0
low_kloss = 100.0

print('\nStarting Training..\n')
info = open(dataname, 'r+') if dataname in os.listdir() else open(dataname, 'x')
 
if not 'output' in os.listdir(): os.mkdir('output')
j = str(sorted(list(map(lambda x: int(x.split('data_')[1]), os.listdir('output'))), reverse=True)[0]+1)\
os.mkdir('./output/data_'+j)

'''
takes too much memory, however can be useful

os.mkdir('./output/data_'+j+'/images'+str(num_im))
os.mkdir('./output/data_'+j+'/all_weights')
'''

timer = time.time()

for epoch in range(num_epochs):
    last_cycle = time.time()
    '''
    low = 100.0
    low_img = Variable()
    '''

    for i, data in enumerate(dataloader, 0):
        iter_beg = time.time()

        netD.zero_grad()

        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)

        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)

        errD_fake = criterion(output, label)
        errD_fake.backward()

        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        optimizerD.step()
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)

        errG = criterion(output, label)
        errG.backward()

        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            tli = (len(dataloader)*num_epochs-len(dataloader)*epoch-i)*time_per_iter
            tle = tle - time.time() + last_cycle
            if epoch == 0: tle *= 0
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            print('time passed: ['+str(int(time.time()-timer))+'s]'
                  +'('+str(int(((time.time()-timer)//60)//60))+'h '
                  +str(int(((time.time()-timer)//60)%60))+'min '
                  +str(int((time.time()-timer)%60))+'s)'
                  +'\ttime after epoch: ['+str(int(time.time()-last_cycle))+'s]'
                  +'('+str(int((time.time()-last_cycle)//60))+'min '+str(int((time.time()-last_cycle)%60))+'s)')
            print('left: [%dh %dmin %dsec]\t(%dh %dmin %dsec)'
                  % (tli//60//60, tli//60%60, tli%60,
                     tle//60//60, tle//60%60, tle%60))

        G_losses.append(errG.item())
        D_losses.append(errD.item())
        loss_time.append(time.time())
        epochs.append(epoch)
        loader.append(i)

        if errD.item() < low_Dloss:
            with torch.no_grad():
                best_Dimage = netG(fixed_noise).detach().cpu()
                low_Dloss = errD.item()
                os.remove('model/dis_model1.pth')
                torch.save(netD.state_dict(),'model/dis_model1.pth')
        if errG.item() < low_Gloss:
            with torch.no_grad():
                best_Gimage = netG(fixed_noise).detach().cpu()
                low_Gloss = errG.item()
                os.remove('model/gen_model1.pth')
                torch.save(netG.state_dict(),'model/gen_model1.pth')
        if (errD.item() + errG.item()) < low_kloss:
            with torch.no_grad():
                best_kimage = netG(fixed_noise).detach().cpu()
                low_kloss = errD.item() + errG.item()
                klossD = errD.item()
                klossG = errG.item()
                os.remove('model/dis_model.pth')
                os.remove('model/gen_model.pth')
                torch.save(netD.state_dict(),'model/dis_model.pth')
                torch.save(netG.state_dict(),'model/gen_model.pth')
        '''
        if (errD.item() + errG.item()) < low:
            with torch.no_grad():
                low = errD.item() + errG.item()
                low_img = netG(fixed_noise).detach().cpu()
                low_1mg = netG(fnoise).detach().cpu()
                low_weight = netG.state_dict()
        '''

        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
        
        del real_cpu
        del b_size
        del label
        del output
        del errD_real
        del errD_fake
        del noise
        del D_x
        del D_G_z1
        del D_G_z2
        del fake
        del errD
        del errG

        time_per_iter = time.time()-iter_beg
    time_per_epoch = time.time()-last_cycle
    tle = (num_epochs-epoch-1)*time_per_epoch
    perepoch.append(time.time()-last_cycle)

    '''
    save_image(vutils.make_grid(low_img, padding=2, normalize=True),
                'output/data_'+j+'/images'+str(num_im)+'/'+str(epoch)+'.png')
    save_image(vutils.make_grid(low_1mg, padding=2, normalize=True),
                'output/data_'+j+'/images/'+str(epoch)+'.png')
    torch.save(low_weight,
               'output/data_'+j+'/all_weights/'+str(epoch)+'.pth')
    '''

print('\nTraining Comlete.\n')

with open('output/data_'+j+'/info.txt', 'w') as file:
    file.write('low Discriminator loss:\t\t'+str(low_Dloss)+'\n')
    file.write('low Generator loss:\t\t'+str(low_Gloss)+'\n')
    file.write('low GD loss:\t\t\t'+str(low_kloss)+'\n')
    file.write('GD Discriminator loss:\t\t'+str(klossD)+'\n')
    file.write('GD Generator loss: \t\t'+str(klossG)+'\n')
    file.write('time passed: ['+str(int(time.time()-timer))+'s]'
               +'('+str(int(((time.time()-timer)//60)//60))+'h '
               +str(int(((time.time()-timer)//60)%60))+'min '
               +str(int((time.time()-timer)%60))+'s)'+'\n')
    file.write('ran epochs: '+str(num_epochs)+'\n')
    file.write('reached low dloss:'+str(saved_iter))
file.close()

with open('output/data_'+j+'/all_losses.txt', 'w') as file:
    for k in range(len(epochs)):
        file.write('[%d/%d](%d/%d)\t{%dh %dmin %dsec}\tLoss_D: %.4f\t Loss_G: %.4f\n'
                   % (epochs[k], num_epochs, loader[k], len(dataloader),
                      loss_time[k]//60//60, loss_time[k]//60%60, loss_time[k]%60,
                      G_losses[k], D_losses[k]))
file.close()

save_image(vutils.make_grid(best_Dimage, padding=2, normalize=True), 'output/data_'+j+'/low_d.png')
save_image(vutils.make_grid(best_Gimage, padding=2, normalize=True), 'output/data_'+j+'/low_g.png')
save_image(vutils.make_grid(best_kimage, padding=2, normalize=True), 'output/data_'+j+'/low_k.png')

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('output/data_'+j+'/loss_graph.png')

plt.figure(figsize=(10,5))
plt.title("Time per epoch")
plt.plot(perepoch,label='T')
plt.xlabel("iterations")
plt.ylabel("Seconds")
plt.legend()
plt.savefig('output/data_'+j+'/time_graph.png')

fig = plt.figure(figsize=(8,8))
plt.axis("off")

ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000, blit=False)
ani.save('output/data_'+j+'/output_ani.gif')
clip = mp.VideoFileClip('output/data_'+j+'/output_ani.gif')
clip.write_videofile('output/data_'+j+'/output_ani.mp4')

if not str(dataroot)+str(image_size) in (lambda x: x.split(sym), (str(info).split('\n'))):
    info.write(str(dataroot)+str(image_size)+sym+str(low_Dloss)+sym+str(low_Gloss)+sym+str(low_kloss)+sym+str(klossD)+sym+str(klossG)+'\n')
    info.close()
else:
    low_loss = lambda x: x.split(sym), (str(info).split('\n'))
    info.close()
    for k in range(len(low_loss)):
        if low_loss[k].split(sym)[0] == str(dataroot)+str(image_size): break
    lis = low_loss[k].split(sym)
    if float(lis[1]) < low_Dloss: lis[1] = str(low_Dloss)
    if float(lis[2]) < low_Gloss: lis[2] = str(low_Gloss)
    if float(lis[3]) < low_kloss: lis[3] = str(low_kloss)
    if float(lis[4]) < low_kloss: lis[3] = str(klossD)
    if float(lis[5]) < low_kloss: lis[3] = str(klossG)
    low_loss[k] = sym.join(lis)
    with open(dataname, 'w') as file:
        file.write(low_loss)
    file.close()
    
    

