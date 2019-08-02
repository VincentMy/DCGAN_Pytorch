from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import time
#from multiprocessing import freeze_support,Pool
"""
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',default='D:/Document/data/CelebA/Img',help='Root diectory for dataset')
parser.add_argument('--workers',type=int,default=2,help='Number of workers for dataloader')
parser.add_argument('--batch_size',type=int,default=128,help='batch size during training')
parser.add_argument('--image_size',type=int,default=64,help='Spatial size of training images.All images.All images will be resized to this size using a transformer')
parser.add_argument('--nc',type=int,default=3,help='number of channels in the training images. For color images this is 3')
parser.add_argument('--nz',type=int,default=100,help='size of z latent vector(i.e. size of generator input)')
parser.add_argument('--ngf',type=int,default=64,help='size of feature maps in generator')
parser.add_argument('--ndf',type=int,default=64,help='size of feature maps in discriminator')
parser.add_argument('--num_epochs',type=int,default=5,help='Number of training epochs')
parser.add_argument('--lr',type=float,default=0.0002,help='Learning rate for optimizers')
parser.add_argument('--beta1',type=float,default=0.5,help='Beta1 hyperparam for Adam optimizers')
parser.add_argument('--ngpu',type=int,default=1,help='Number of GPUs available.Use 0 for CPU mode')
parser.add_argument('--manualSeed',type=int,default=999,help='manual seed')
opt = parser.parse_args()
"""
# Root directory for dataset
dataroot = "D:/Document/data/CelebA/Img"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 3

# Learning rate for optimizers
#lr = 0.0002
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

save_pathG = 'D:/PythonWorks/DCGAN/checkout/parameterG.pkl'
save_pathD = 'D:/PythonWorks/DCGAN/checkout/parameterD.pkl'
save_pathA = 'D:/PythonWorks/DCGAN/checkout/parameter_1600.pkl'
save_path = 'D:/PythonWorks/DCGAN/checkout/'

manualSeed = 999
#manualSeed = random.randint(1,1000) #use if you want new result
print("Random Seed:",manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
#We can use an image folder dataset the way we have it setup
#Create the dataset

dataset = dset.ImageFolder(root=dataroot,transform=transforms.Compose([
    transforms.Resize(image_size),transforms.CenterCrop(image_size),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),]))
#Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=workers)
#Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def draw(dataloader):
    #Plot some training images
    real_batch = next(iter(dataloader))
    print(len(real_batch))
    #print(real_batch)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],padding=2,normalize=True).cpu(),(1,2,0)))
    plt.show()
#custom weights initialization called on netG and netD

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)

#Generator Code

class Generator(nn.Module):
    def __init__(self,ngpu):
        super(Generator,self).__init__()
        self.ngpu = ngpu
        
        self.main = nn.Sequential(
            #input is Z going into a convolution
            #in_channels,out_channels,kernel_size,stride
            nn.ConvTranspose2d(nz,ngf*8,4,1,0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        
    def forward(self,input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self,ngpu):
        super(Discriminator,self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x64 x64
            nn.Conv2d(nc,ndf,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf,ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self,input):
        return self.main(input)


netG = Generator(ngpu).to(device)
if(device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG,list(range(ngpu)))
#Apply the weight_init function to randomly initialize all weights
#to mean=0,stdev=0.2
netG.apply(weights_init)
netA_CKPT = torch.load(save_pathA)
netG.load_state_dict(netA_CKPT['state_dictG'])
#print(netG)
netD = Discriminator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD,list(range(ngpu)))
#Apply the weights_init function to randomly initialize all weights
#to mean=0,stdev=0.2
netD.apply(weights_init)
netD.load_state_dict(netA_CKPT['state_dictD'])
#netD = torch.load(save_pathD)
#print(netD)
#Initialize
criterion = nn.BCELoss()
#Create batch of latent vectors that we will use to visulize
#the progression of the generator
fixed_noise = torch.randn(64,nz,1,1,device=device)
#Establish convention for real and fake labels during training
real_label=1
fake_label=0
#Stup Adam optimizers for both G and  D
optimizerD = optim.Adam(netD.parameters(),lr=lr,betas=(beta1,0.999))
optimizerG = optim.Adam(netG.parameters(),lr=lr,betas=(beta1,0.999))
optimizerG.load_state_dict(netA_CKPT['optimizerG_state_dict'])
optimizerD.load_state_dict(netA_CKPT['optimizerD_state_dict'])
#Training Loop
#Lists to keep track of progress
img_list=[]
G_losses=[]
D_losses=[]

print("Starting Training Loop...")
def train_net():
    iters=0
    #For each epoch
    for epoch in range(num_epochs):
        
        for i,data in enumerate(dataloader,0):
            #(1):Update D network:maximize log(D(x)) +log(1-D(G(z)))
            #Train with all-real batch,data :torch.Size([128, 3, 64, 64])
            #每次提取出的data维度为([128, 3, 64, 64])
            netD.zero_grad()
            #Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,),real_label,device=device)
            #Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            #Calculate loss on all-real batch
            errD_real=criterion(output,label)
            #Calculate gradients for D in backward pass
            errD_real.backward()
            #输出的均值，应该趋于1
            D_x=output.mean().item()
            #Train with all-fake batch
            #Generate batch of latent vectors
            noise = torch.randn(b_size,nz,1,1,device=device)
            #Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            #Classify all fake batch wiht D
            output=netD(fake.detach()).view(-1)
            #Calculate D`s loss on the all-fake batch
            errD_fake=criterion(output,label)
            #Calculate the gradients for this batch
            errD_fake.backward()
            #生成网络的输出值均值，应该趋于0
            D_G_z1 = output.mean().item()
            #Add the gradients from the all-real and all-fake batches
            #判别网络总的损失值
            errD = errD_real + errD_fake
            #Update D
            optimizerD.step()
            #(2):Update G network:maximize log(D(G(z)))
            netG.zero_grad
            #fake labels are real for generator cost
            label.fill_(real_label)
            #Since we just update D,perform another forward pass of all-fake batch through D
            #更新梯度后，向生成网络中输入随机数
            output=netD(fake).view(-1)
            #Calculate G`s loss based on this output，判断生成网络生成的图片和真实label之间的损失函数
            errG=criterion(output,label)
            #Calculate gradients for G
            errG.backward()
            #通过一次梯度更新后，生成网络D输出的均值
            D_G_z2=output.mean().item()
            #Update G
            optimizerG.step()
            #output training stats
            if i % 50 ==0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'% (epoch, 
                    num_epochs, i, len(dataloader),errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            #Save losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            #Check how the generator is doing by saving G`s output on fixed_noise
            if(iters % 500 == 0) or ((epoch==num_epochs-1) and (i==len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake,padding=2,normalize=True))
                print("img_list:",len(img_list))
            iters += 1
            if(iters % 10 == 0):
                print("iters:",iters)
            if(iters % 100 == 0 or iters == len(dataloader) - 1):
                new_save_path = os.path.join(save_path,'parameter2_'+ str(iters) + '.pkl')
                print(new_save_path)
                if not os.path.exists(new_save_path):
                    file = open(new_save_path,'w')
                    file.close()
                    #time.sleep(1)
                torch.save({
                    'state_dictG':netG.state_dict(),
                    'state_dictD':netD.state_dict(),
                    'optimizerG_state_dict':optimizerG.state_dict(),
                    'optimizerD_state_dict':optimizerD.state_dict()
                },new_save_path)
                #print("iters:",iters)
def draw_loss():
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def draw_picf():
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)),animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig,ims,interval=1000,repeat_delay=1000,blit=True)
    HTML(ani.to_jshtml())
def draw_comp():
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()

def main():
    train_net()
    print(len(img_list))
    #draw_loss()
    #draw_picf()
    draw_comp()
    #printG()
    #printD()
if __name__ == "__main__":
    main()

