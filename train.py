import os
import numpy as np
import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch import autograd
import time
import matplotlib.pyplot as plt
import itertools

import config
import utils
import models


def GP(D, real, fake, config, device):
    alpha = torch.rand(config.batch_size, config.channel, 1, 1)
    alpha = alpha.expand(real.size())
    alpha = alpha.float().to(device)
    
    xhat = alpha * real + (1-alpha) * fake
    xhat = xhat.float().to(device)
    xhat = autograd.Variable(xhat, requires_grad = True)
    xhat_D = D(xhat)
    
    grad = autograd.grad(outputs=xhat_D, inputs=xhat, grad_outputs=torch.ones(xhat_D.size()).to(device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean() * config.eta
    
    return penalty


def train(config, device, A, B):
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    
    GA2B = models.Generator(config)
    DA = models.Discriminator(config)
    GB2A = models.Generator(config)
    DB = models.Discriminator(config)
    

    
    DA.train()
    DB.train()
    
    if torch.cuda.device_count() > 1 and device=='cuda':
        print("GPU count : {}".format(torch.cuda.device_count()))
        print(device)
        GA2B = nn.DataParallel(GA2B).cuda()
        DA = nn.DataParallel(DA).cuda()
        GB2A = nn.DataParallel(GB2A).cuda()
        DB = nn.DataParallel(DB).cuda()
    elif torch.cuda.device_count() == 1 and device=='cuda':
        GA2B = nn.DataParallel(GA2B).cuda()
        DA = nn.DataParallel(DA).cuda()
        GB2A = nn.DataParallel(GB2A).cuda()
        DB = nn.DataParallel(DB).cuda()
        
        

    
    print(GA2B)
    print(DA)
    
    print("Training start ...")
    
    epochD_lossA = []
    epochD_lossB = []
    epochG_lossA = []
    epochG_lossB = []
    for e in range(config.epochs):
        
        lossesA = []
        lossesB = []

        print("Epoch {} started ...".format(e+1))
        print("Training Discriminator ... Intermediate result can be found in "+config.result_dir)
        
        G_lr = config.G_lr
        D_lr = config.D_lr
        
        if config.lr_decay=='linear' and config.decay_epoch<(e+1):
            print("Learning rate decay ... Option : {}".format(config.lr_decay))
            G_lr = config.G_lr * ((e+1)/config.epochs)
            D_lr = config.D_lr * ((e+1)/config.epochs)
        
        optimizer_G = optim.Adam(itertools.chain(GA2B.parameters(), GB2A.parameters()), lr=G_lr, betas=(config.beta1, config.beta2))
        optimizer_D = optim.Adam(itertools.chain(DA.parameters(), DB.parameters()), lr=D_lr, betas=(config.beta1, config.beta2))
        
        for D_idx in tqdm.tqdm(range(min((config.D_iter // config.batch_size) * (e+1), config.D_max_iter // config.batch_size))):
            
            optimizer_D.zero_grad()

            batch_A = np.random.choice(A, config.batch_size, replace=False)
            batch_B = np.random.choice(B, config.batch_size, replace=False)
            batch_A = utils.preprocessing(batch_A, config)
            batch_B = utils.preprocessing(batch_B, config)
            batch_A = torch.from_numpy(batch_A).float().to(device)
            batch_B = torch.from_numpy(batch_B).float().to(device)
            
            
            fake_A2B = GA2B(batch_A)
            fake_B2A = GB2A(batch_B)

            D_real_loss_A = DA(batch_A).mean()
            D_fake_loss_A = DA(fake_B2A).mean()

            D_real_loss_B = DB(batch_B).mean()
            D_fake_loss_B = DB(fake_A2B).mean()

            GP_A = GP(DA, batch_A, fake_B2A, config, device)
            GP_B = GP(DB, batch_B, fake_A2B, config, device)

            loss_OTDisc_A = -D_real_loss_A + D_fake_loss_A + GP_A
            loss_OTDisc_B = -D_real_loss_B + D_fake_loss_B + GP_B

            loss_OTDisc_A.backward(retain_graph=True)
            loss_OTDisc_B.backward(retain_graph=True)

            optimizer_D.step()

            lossesA.append(loss_OTDisc_A.data.cpu().numpy())
            lossesB.append(loss_OTDisc_B.data.cpu().numpy())

            if (D_idx+1) % config.print_step == 0:
                plt.ylim(-1,1)
                plt.plot(lossesA)
                plt.savefig(os.path.join(config.result_dir, 'D_lossesA_for_epoch_{}.png'.format(e+1)))
                plt.close('all')
                plt.ylim(-1,1)
                plt.plot(lossesB)
                plt.savefig(os.path.join(config.result_dir, 'D_lossesB_for_epoch_{}.png'.format(e+1)))
                plt.close('all')
                
        epochD_lossA.append(np.mean(lossesA))
        epochD_lossB.append(np.mean(lossesB))
        plt.ylim(-1,1)
        plt.plot(epochD_lossA)
        plt.savefig(os.path.join(config.result_dir, 'D_loss_per_epochA.png'))
        plt.close('all')
        plt.ylim(-1,1)
        plt.plot(epochD_lossB)
        plt.savefig(os.path.join(config.result_dir, 'D_loss_per_epochB.png'))
        plt.close('all')
        
        
        lossesA = []
        lossesB = []
        
        print("Training Generator ... Intermediate result can be found in "+config.result_dir)
        
        for G_idx in tqdm.tqdm(range((config.G_iter // config.batch_size))):
            optimizer_G.zero_grad()
            
            batch_A = np.random.choice(A, config.batch_size, replace=False)
            batch_B = np.random.choice(B, config.batch_size, replace=False)
            batch_A = utils.preprocessing(batch_A, config)
            batch_B = utils.preprocessing(batch_B, config)
            batch_A = torch.from_numpy(batch_A).float().to(device)
            batch_B = torch.from_numpy(batch_B).float().to(device)

            fake_A2B = GA2B(batch_A)
            fake_B2A = GB2A(batch_B)
            
            recon_A2A = GB2A(fake_A2B)
            recon_B2B = GA2B(fake_B2A)
            
            cyclic_loss_A = torch.abs(batch_A - recon_A2A).mean()
            cyclic_loss_B = torch.abs(batch_B - recon_B2B).mean()
            
            D_real_loss_A = DA(batch_A).mean()
            D_fake_loss_A = DA(fake_B2A).mean()

            D_real_loss_B = DB(batch_B).mean()
            D_fake_loss_B = DB(fake_A2B).mean()

            GP_A = GP(DA, batch_A, fake_B2A, config, device)
            GP_B = GP(DB, batch_B, fake_A2B, config, device)
            
            loss_OTDisc_A = -D_real_loss_A + D_fake_loss_A + GP_A
            loss_OTDisc_B = -D_real_loss_B + D_fake_loss_B + GP_B
            
            GA_loss = config.gamma * cyclic_loss_A - loss_OTDisc_A
            GB_loss = config.gamma * cyclic_loss_B - loss_OTDisc_B
            
            GA_loss.backward(retain_graph=True)
            GB_loss.backward(retain_graph=True)
            
            optimizer_G.step()

            lossesA.append(GA_loss.data.cpu().numpy())
            lossesB.append(GB_loss.data.cpu().numpy())

            if (G_idx+1) % config.print_step == 0:
                plt.ylim(-1,1)
                plt.plot(lossesA)
                plt.savefig(os.path.join(config.result_dir, 'G_lossesA_for_epoch_{}.png'.format(e+1)))
                plt.close('all')
                plt.ylim(-1,1)
                plt.plot(lossesB)
                plt.savefig(os.path.join(config.result_dir, 'G_lossesB_for_epoch_{}.png'.format(e+1)))
                plt.close('all')
                
        epochG_lossA.append(np.mean(lossesA))
        epochG_lossB.append(np.mean(lossesB))
        plt.ylim(-1,1)
        plt.plot(epochG_lossA)
        plt.savefig(os.path.join(config.result_dir, 'G_loss_per_epochA.png'))
        plt.close('all')
        plt.ylim(-1,1)
        plt.plot(epochG_lossB)
        plt.savefig(os.path.join(config.result_dir, 'G_loss_per_epochB.png'))
        plt.close('all')
        
        print("Saving Generator at {}".format(config.checkpoint_dir) + " ...")
        
        torch.save(GA2B.state_dict(), os.path.join(config.checkpoint_dir, 'modelA2B_{:04d}'.format(e+1) + '.pth'))
        torch.save(GB2A.state_dict(), os.path.join(config.checkpoint_dir, 'modelB2A_{:04d}'.format(e+1) + '.pth'))
        
            
def main():
    
    c = config.configuration()
    print(c)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ['CUDA_VISIBLE_DEVICES'] = c.gpu

    list_A, list_B = utils.get_data(c)

    print("Number of data for domain A : {}".format(len(list_A)))
    print("Number of data for domain B : {}".format(len(list_B)))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train(c, device, list_A, list_B)
    
    print("Exiting ...")
    
if __name__ == '__main__':
    main()

# python train.py
# python train.py --D_iter 300 --result_dir results_identity_biasFalse --checkpoint_dir checkpoints_identity_biasFalse --gpu 0.1 --batch_size 2
# python train.py --D_iter 300 --result_dir results_identity_biasFalse --checkpoint_dir checkpoints_identity_biasFalse --gpu 0,1 --batch_size 2
# python train.py --D_iter 300 --result_dir results_convtranspose_biasFalse --checkpoint_dir checkpoints_convtranspose_biasFalse --gpu 0,1 --batch_size 2
# python train.py --D_iter 300 --result_dir results_convtranspose_Gfirst --checkpoint_dir checkpoints_convtranspose_Gfirst --gpu 0,1 --batch_size 2