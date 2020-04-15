#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:16:12 2020

@author: zhantao
"""

import torch
from torch import optim, nn
import matplotlib.pyplot as plt

from dl_helper import load_data
from models import variantAutoencoder, visualize, train

'''

Support modes:
    
    autoEncoder, autoEncoderLargercodes :  training [ noise22noise, denoise, None ]
                                           visualize [ simple, denoise ]
    
    superResolution, superResolutiionLargercodes : training [ superresolution ]
                                                   visualize[ superresolution ] 
                                                       
    variantAutoencoder, variantAutoencoderLarger : training [ None ]
                                                   visualize[ simple, denoise ]
'''

# In[]
# train networks and visualize it performance

numEpoch  = 200
batchSize = 100
stepSize  = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# prepare for training
train_input, _, test_input, _ = load_data(one_hot_labels = True, normalize = True, flatten = False)

model = variantAutoencoder(channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr = stepSize)
criterion = nn.MSELoss() # 'mean' is elementwise mean, which is not what we want

# start training
model, lossList = train(model, optimizer, criterion, train_input, device, numEpoch, batchSize, mode = 'denoise', noise_scale = 1)

plt.plot(lossList)
plt.grid()

vis = visualize(model, ['denoise'], test_input, device, noise_scale = 1, sample_index = [1,456,23,876,234])
vis()


# In[]
# visualiza the transformation between numbers and the learned featrures

# transformation 
index0, index1 = 2, 12
feature0 = model.encoder(test_input[index0,0][None,None,:,:].to(device))
feature1 = model.encoder(test_input[index1,0][None,None,:,:].to(device))

synImgList = []
for weight in torch.arange(0, 1, 0.1):
    
    syn_feature = feature0 * weight + feature1 * (1 - weight)
    
    if model.name == 'variantAE':
        synImgList.append(model.decoder(syn_feature[:, :syn_feature.size(1)//2])[0,0].detach().to('cpu'))
    else:
        synImgList.append(model.decoder(syn_feature)[0,0].detach().to('cpu'))

visImg = torch.cat( (test_input[index1,0], torch.cat(synImgList, dim=1), test_input[index0,0]), dim = 1 )
plt.figure()
plt.imshow(visImg)
plt.title('transformation')



# learned features
numSteps = 30
reconstructedImages = []
MAXfeature = feature0.max().item()*3
MINfeature = feature0.min().item()*3

if model.name == 'variantAE':
    feature = feature0[:, :feature0.size(1)//2] 
    for ind in range(feature.shape[1]):
        changedOneFeature = []
        for disturb in torch.arange(MINfeature, MAXfeature, (MAXfeature-MINfeature)/numSteps):
            disturbance = torch.zeros_like(feature)
            disturbance[:,ind,:,:] += disturb
            changedOneFeature.append(model.decoder(feature + disturbance).detach().to('cpu')[0,0])
        reconstructedImages.append( torch.cat(changedOneFeature, dim = 1) )
else: 
    for ind in range(feature0.shape[1]):
        changedOneFeature = []
        for disturb in torch.arange(MINfeature, MAXfeature, (MAXfeature-MINfeature)/numSteps):
            disturbance = torch.zeros_like(feature0)
            disturbance[:,ind,:,:] += disturb 
            changedOneFeature.append(model.decoder(feature0 + disturbance).detach().to('cpu')[0,0])
        reconstructedImages.append( torch.cat(changedOneFeature, dim = 1) )

plt.figure(figsize = [12,8])
plt.imshow( torch.cat(reconstructedImages, dim = 0), cmap = 'gray')
plt.title('learned features')
        


# targeting 2 features
MIN_, MAX_, numSteps= -3, 3, 15
reconstructedImages  = []
featureInd0, featureInd1 = 0, 5

row, col = torch.arange(MIN_, MAX_, (MAX_-MIN_)/numSteps), torch.arange(MIN_, MAX_, (MAX_-MIN_)/numSteps)
rowgrid, colgrid = torch.meshgrid(row, col)
disturbanceGrid  = torch.stack((rowgrid, colgrid), dim = 2).reshape(-1,2)

if model.name == 'variantAE':
    feature = feature0[:, :feature0.size(1)//2] 
    for ind in range(disturbanceGrid.shape[0]):
        disturbance = torch.zeros_like(feature)
        disturbance[:,featureInd0,:,:] += disturbanceGrid[ind,0]
        disturbance[:,featureInd1,:,:] += disturbanceGrid[ind,1]
        
        reconstructedImages.append(model.decoder(feature + disturbance).detach().to('cpu')[0,0])
else: 
    for ind in range(disturbanceGrid.shape[0]):
        disturbance = torch.zeros_like(feature0)
        disturbance[:,featureInd0,:,:] += disturbanceGrid[ind,0]
        disturbance[:,featureInd1,:,:] += disturbanceGrid[ind,1]
        
        reconstructedImages.append(model.decoder(feature0 + disturbance).detach().to('cpu')[0,0])

plt.figure(figsize = [12,8])
plt.imshow(torch.stack(reconstructedImages, dim = 0).contiguous().view(numSteps, 28*numSteps, -1)
           .permute(1, 0, 2).contiguous().view(28*numSteps, 28*numSteps), cmap = 'gray')
plt.title('learned features : %.d and %.d'%(featureInd0, featureInd1))
