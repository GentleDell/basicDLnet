#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:58:15 2020

@author: zhantao
"""

import torch
from torch import nn
from torch.nn import functional as F

import matplotlib.pyplot as plt

class autoEncoder(nn.Module):
    '''
    It implements a simple autoencoder. Convolutional layer plus a relu 
    activation function make up of the basic unit of the autoencoder.
    
    We use this structure as a benchmark to compare with other networks,
    such as denoising, VAE and etc.
    
    To improve its performance, increasing the codes dimension is the most 
    effective choice. Current loss with code = 8*1*1, 200 epochs is ~0.126.
    After changing the codes to 12*3*3, the loss decrease to ~0.03. Further 
    decrease could be achieved if we continu to increase the size. But then 
    the autoencoder does not keep only the essence of minst dataset.
        
    '''
    def __init__(self, channels : int):
        super(autoEncoder, self).__init__()
        
        self.name  = 'autoencoder'
        
        self.conv1 = nn.Conv2d(channels , 32, kernel_size=(5, 5), stride=(1, 1))   # 32*24*24
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1))          # 32*20*20
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(4, 4), stride=(2, 2))          # 32*9*9
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2))          # 32*4*4
        self.conv5 = nn.Conv2d(32, 8 , kernel_size=(4, 4), stride=(1, 1))          # 8*1*1
        
        self.tconv1 = nn.ConvTranspose2d(8 , 32, kernel_size=(4, 4), stride=(1, 1))    # 32*4*4
        self.tconv2 = nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2))    # 32*9*9
        self.tconv3 = nn.ConvTranspose2d(32, 32, kernel_size=(4, 4), stride=(2, 2))    # 32*20*20
        self.tconv4 = nn.ConvTranspose2d(32, 32, kernel_size=(5, 5), stride=(1, 1))    # 32*24*24
        self.tconv5 = nn.ConvTranspose2d(32, channels , kernel_size=(5, 5), stride=(1, 1))  # 32*28*28

    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x

    def decoder(self, x):
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x))
        x = self.tconv5(x)
        return x

    def forward(self, x):
        
        codes = self.encoder(x)
        recon = self.decoder(codes)
        
        return recon
    
    
class autoEncoderLargercodes(autoEncoder):
    '''
    It implements a simple autoencoder with larger codes to compare with the 
    benchmark. Convolutional layer plus a relu activation function make up of 
    the basic unit of the autoencoder.
    
    From the results we can see that, after increase the dim and size of the 
    codes, it performs much better than the benchmark autoencoder in tasks of 
    denoising. But if the noise is much larger than the training noise, it 
    fails fast as the benchmark.
        
    '''
    def __init__(self, channels : int):
        super(autoEncoderLargercodes, self).__init__(channels)
        
        self.name  = 'autoencoder'
        
        self.conv5 = nn.Conv2d(32, 8 , kernel_size=(2, 2), stride=(1, 1))              # 8*3*3,
        self.tconv1 = nn.ConvTranspose2d(8 , 32, kernel_size=(2, 2), stride=(1, 1))    # 32*4*4
        

    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x

    def decoder(self, x):
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x))
        x = self.tconv5(x)
        return x

    def forward(self, x):
        
        codes = self.encoder(x)
        recon = self.decoder(codes)
        
        return recon

    
class superResolution(nn.Module):
    '''
    It implements a simple 2x superresolution autoencoder for the MINST set.
    It is used as a benchmark.
    '''
    def __init__(self, channels : int):
        super(superResolution, self).__init__()
        
        self.name  = 'superresolution'
        
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=(5, 5), stride=(1, 1))    # 32*10*10
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1))          # 32*6*6
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(4, 4), stride=(1, 1))          # 32*3*3
        self.conv4 = nn.Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1))           # 8*1*1
        
        self.tconv1 = nn.ConvTranspose2d(8 , 32, kernel_size=(4, 4), stride=(1, 1))    # 32*4*4
        self.tconv2 = nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2))    # 32*9*9
        self.tconv3 = nn.ConvTranspose2d(32, 32, kernel_size=(4, 4), stride=(2, 2))    # 32*20*20
        self.tconv4 = nn.ConvTranspose2d(32, 32, kernel_size=(5, 5), stride=(1, 1))    # 32*24*24
        self.tconv5 = nn.ConvTranspose2d(32, channels , kernel_size=(5, 5), stride=(1, 1))  # 32*28*28
        
    def encoder(self, x):
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

    def decoder(self, x):
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x))
        x = self.tconv5(x)
        return x

    def forward(self, x):
        
        codes = self.encoder(x)
        recon = self.decoder(codes)
        
        return recon
    

class superResolutiionLargercodes(superResolution):
    '''
    It implements a simple 2x superresolution autoencoder for the MINST set
    with larger codes to compare with the benchmark superresolution network. 
    
    From the results we can see that, after increase the dim and size of the 
    codes, it performs much better than the benchmark superresolution network.        
    '''
    def __init__(self, channels : int):
        super(superResolutiionLargercodes, self).__init__(channels)
        
        self.name  = 'superresolution'
        
        self.conv4 = nn.Conv2d(32, 8 , kernel_size=(1, 1), stride=(1, 1))              # 8*3*3,
        self.tconv1 = nn.ConvTranspose2d(8 , 32, kernel_size=(2, 2), stride=(1, 1))    # 32*4*4
        
    def encoder(self, x):
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

    def decoder(self, x):
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x))
        x = self.tconv5(x)
        return x

    def forward(self, x):
        
        codes = self.encoder(x)
        recon = self.decoder(codes)
        
        return recon
    
    
class variantAutoencoder(autoEncoder):
    '''
    It implements a simple variant autoendcoder by add kl loss to the codes
    and use the code to do random sampling for decoding.
    
    This VAE is based on the benchmark autoencoder. 
    '''
    def __init__(self, channels : int):
       super(variantAutoencoder, self).__init__(channels)     
       self.name = 'variantAE'
       self.conv5 = nn.Conv2d(32, 16 , kernel_size=(4, 4), stride=(1, 1))          # 16*1*1 = (mu, var)
     
    def forward(self, x):
     
       param_f = self.encoder(x)
       
       mu_f, logvar_f = param_f.split(param_f.size(1)//2, 1)
     
       kl = -0.5 * (1 + logvar_f - mu_f.pow(2) - logvar_f.exp())
       kl_loss  = kl.sum() / x.size(0)
     
       std_f = torch.exp(0.5 * logvar_f)
       codes = torch.empty_like(mu_f).normal_() * std_f + mu_f
     
       recon = self.decoder(codes)
     
       return recon, kl_loss
    

class variantAutoencoderLarger(autoEncoder):
    '''
    It implements a simple variant autoendcoder but using a larger size and 
    more dimension of codes.
    
    This VAE is based on the benchmark autoencoder. 
    
    From the results we can see that, after increasing the dim and size of the 
    codes, it recivers better than the benchmark VAE.  
    '''
    def __init__(self, channels : int):
      super(variantAutoencoderLarger, self).__init__(channels)
      
      self.name = 'variantAE'
      self.conv5 = nn.Conv2d(32, 16 , kernel_size=(2, 2), stride=(1, 1))             # 16*3*3 = (mu, var)
      self.tconv1 = nn.ConvTranspose2d(8 , 32, kernel_size=(2, 2), stride=(1, 1))    # 32*4*4
       
    def forward(self, x):
      
      param_f = self.encoder(x)
      
      mu_f, logvar_f = param_f.split(param_f.size(1)//2, 1)
      
      kl = -0.5 * (1 + logvar_f - mu_f.pow(2) - logvar_f.exp())
      kl_loss  = kl.sum() / x.size(0)
      
      std_f = torch.exp(0.5 * logvar_f)
      codes = (torch.empty_like(mu_f).normal_() * std_f + mu_f)
      
      recon = self.decoder(codes)
      
      return recon, kl_loss


class visualize(nn.Module):
    
    def __init__(self, model, testname, dataset, device, **kwargs):
        super(visualize, self).__init__()
        
        self.model = model
        self.test  = testname
        self.args  = kwargs
        self.data  = dataset
        self.device= device
    
    def superResolution(self):
        index = torch.randint(0, self.data.shape[0], (5,))
        
        if 'sample_index' in self.args: 
            index = self.args['sample_index']
        
        originImg = self.data[index,:,:,:]
        downsampl = F.avg_pool2d(originImg, kernel_size = 2)
        superResl = self.model(downsampl.to(self.device)).detach().to('cpu')
        
        dowsamvis = torch.zeros(superResl.shape)
        dowsamvis[:, :, 7:21, 7:21] = downsampl
        
        visImage  = torch.cat((originImg[:,0,:,:].reshape([-1,28]), 
                               dowsamvis[:,0,:,:].reshape([-1,28]),
                               superResl[:,0,:,:].reshape([-1,28])), dim=1)
        plt.figure()
        plt.imshow(visImage, cmap='gray')
        plt.title('original, reconstructed')
    
    
    def simple(self):
        index = torch.randint(0, self.data.shape[0], (5,))
        
        if 'sample_index' in self.args: 
            index = self.args['sample_index']
        
        originImg = self.data[index,:,:,:]
        
        if self.model.name == 'variantAE':
            reconstct = self.model(originImg.to(self.device))[0].detach().to('cpu')    
        else:
            reconstct = self.model(originImg.to(self.device)).detach().to('cpu')
        
        visImage  = torch.cat((originImg[:,0,:,:].reshape([-1,28]), 
                               reconstct[:,0,:,:].reshape([-1,28])), dim=1)
        plt.figure()
        plt.imshow(visImage, cmap='gray')
        plt.title('original, reconstructed')
        
    
    def denoise(self):
        
        index, scale = torch.randint(0, self.data.shape[0], (5,)), 1.5
        
        if 'noise_scale' in self.args:
            scale = self.args['noise_scale']
        if 'sample_index' in self.args: 
            index = self.args['sample_index']
        
        originImg = self.data[index,:,:,:]
        noisedImg = originImg + torch.randn_like(self.data[index,:,:,:])*scale
        
        if self.model.name == 'variantAE':
            denoised = self.model(noisedImg.to(self.device))[0].detach().to('cpu')
        else:
            denoised  = self.model(noisedImg.to(self.device)).detach().to('cpu')
        
        visImage  = torch.cat((originImg[:,0,:,:].reshape([-1,28]), 
                               noisedImg[:,0,:,:].reshape([-1,28]),
                               denoised [:,0,:,:].reshape([-1,28])), dim=1)
        plt.figure()
        plt.imshow(visImage, cmap='gray')
        plt.title('original, noisy, denoised')
               
    
    def forward(self):
        
        if 'superresolution' in self.test:
            self.superResolution()
            
        else:
            self.simple()
            
            if 'denoise' in self.test:
                self.denoise()
        
        return 0
            
    
def train(model, optimizer, criterion, trainSet, device, numEpoch, batchSize, mode : str = None, **kwargs):
    
    lossList = []
    
    scale = 0.5
    if 'noise_scale' in kwargs:
        scale = kwargs['noise_scale']
    
    for epoch in range(numEpoch):
    
        print('\nepoch: ', epoch)
        for batch in range(0, trainSet.shape[0], batchSize):
        
            data = trainSet.narrow(0, batch, batchSize).to(device)
            
            if model.name == 'variantAE':
                
                output, kl_loss = model.forward(data)
                
                # MSELoss() computes elementwise mean by default. But since
                # the kl loss computes half sample-wise mean, here we should 
                # convert the MSELoss to sample-wise mean so that they have 
                # similar definition and order.
                fit_loss = criterion(output, data)*data.size(-2)*data.size(-1)/2
                loss = kl_loss + fit_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            else:
                
                if mode == 'noise2noise' or model == 'denoise':
                    code = model.encoder( data + torch.randn_like(data)*scale )
                elif mode == 'superresolution':
                    code = model.encoder( F.avg_pool2d(data, kernel_size = 2) )
                else:  
                    code = model.encoder(data)
                    
                output = model.decoder(code)
                
                if mode == 'noise2noise':
                    loss = criterion( output, data + torch.randn_like(data)*scale )
                else:
                    loss = criterion(output, data)
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print('training phase: num of batch {%.d}, loss {%.4f}'%(batch/batchSize, loss.detach()))
    
            lossList.append(loss.detach())
    
    return model, lossList
        