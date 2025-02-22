from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from model import AEModel
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import time
import os
from utils import *
import ipdb
from tqdm import tqdm


def ae_loss(model, x):
    """ 
    TODO 2.1.2: fill in MSE loss between x and its reconstruction. 
    return loss, {recon_loss = loss} 
    """

    # ipdb.set_trace()
    # recon_loss = (model.decoder(model.encoder(x)) - x[:128]) ** 2
    recon_loss = (model.decoder(model.encoder(x)) - x) ** 2
    recon_loss = recon_loss.sum(dim=list(range(len(recon_loss.shape)))[1:])
     # bce?
    recon_loss = recon_loss.mean()

    loss = recon_loss
    
    return loss, OrderedDict(recon_loss=loss)

def vae_loss(model, x, beta = 1):
    """TODO 2.2.2 : Fill in recon_loss and kl_loss. """

    # ipdb.set_trace()
    mu, logstd = model.encoder(x)
    # eps = torch.randn((1,)).item() # logstd.shape)
    eps = torch.randn((x.shape[0],1)).cuda() # logstd.shape)
    #(torch.zeros_like(logstd), torch.ones_like(logstd))
    # ipdb.set_trace()
    sample = mu + eps * logstd.exp().unsqueeze(0)#.pow(2)
    # recon_loss = (model.decoder(model.encoder(x)) - x) ** 2
    # recon_loss = (model.decoder(mu) - x[:128]) ** 2


    # recon_loss = (model.decoder(sample) - x[:128]) ** 2
    recon_loss = (model.decoder(sample) - x) ** 2


    # ipdb.set_trace()
    recon_loss = recon_loss.sum(dim=list(range(len(recon_loss.shape)))[1:])
     # bce?
    recon_loss = recon_loss.mean()

    # encode = model.encoder(x)
    # kl_loss = - .5 * (1. + encode.var().log() + encode.mean() ** 2 - encode.var().log().exp())

    # kl_loss = - .5 * (1. + logstd + mu ** 2 - logstd.exp())
    kl_loss =  .5 * (- 1. + logstd.exp().pow(2) + mu.pow(2) - 2*logstd)
    # ipdb.set_trace()
    kl_loss = kl_loss.sum(dim=list(range(len(kl_loss.shape)))[1:])
    # kl_loss = kl_loss.sum()
    kl_loss = kl_loss.mean()

    total_loss = recon_loss + beta*kl_loss

    return total_loss, OrderedDict(recon_loss=recon_loss, kl_loss=kl_loss)


def constant_beta_scheduler(target_val = 1):
    def _helper(epoch):
        return target_val
    return _helper

def linear_beta_scheduler(max_epochs=None, target_val = 1):
    """TODO 2.3.2 : Fill in helper. The value returned should increase linearly 
    from 0 at epoch 0 to target_val at epoch max_epochs """
    def _helper(epoch):
        return target_val * epoch / max_epochs 
    #    ...
    return _helper

def run_train_epoch(model, loss_mode, train_loader, optimizer, beta = 1, grad_clip = 1):
    model.train()
    all_metrics = []
    # for x, _ in tqdm(list(train_loader)[:195]):
    for x, _ in tqdm(train_loader):
        # print("looping")
        x = preprocess_data(x)
        # print(x.shape)
        # ipdb.set_trace()
        # if tuple(x.size()) != (256,3,32,32):
            # continue
            # ipdb.set_trace()
        if loss_mode == 'ae':
            loss, _metric = ae_loss(model, x)
        elif loss_mode == 'vae':
            loss, _metric = vae_loss(model, x, beta)
        all_metrics.append(_metric)
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        # print(all_metrics)

    # ipdb.set_trace()
    return avg_dict(all_metrics)


def get_val_metrics(model, loss_mode, val_loader):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        # for x, _ in tqdm(list(val_loader)[:195]):
        for x, _ in tqdm(val_loader):
            x = preprocess_data(x)
            # print(x.shape)
            # if tuple(x.size()) != (256,3,32,32):
                # ipdb.set_trace()
                # continue
            if loss_mode == 'ae':
                _, _metric = ae_loss(model, x)
            elif loss_mode == 'vae':
                _, _metric = vae_loss(model, x)
            all_metrics.append(_metric)

    return avg_dict(all_metrics)

def main(log_dir, loss_mode = 'vae', beta_mode = 'constant', num_epochs = 20, batch_size = 256, latent_size = 256,
         target_beta_val = 1, grad_clip=1, lr = 1e-3, eval_interval = 5):

    os.makedirs('data/'+ log_dir, exist_ok = True)
    train_loader, val_loader = get_dataloaders()

    variational = True if loss_mode == 'vae' else False
    model = AEModel(variational, latent_size, input_shape = (3, 32, 32)).to(device = ("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # _, new_val_loader = get_dataloaders()
    # vis_x = list(new_val_loader)[0][:36]
    vis_x = next(iter(val_loader))[0][:36]
    # vis_x = next(iter(val_loader))[0][:36]
    
    #beta_mode is for part 2.3, you can ignore it for parts 2.1, 2.2
    if beta_mode == 'constant':
        beta_fn = constant_beta_scheduler(target_val = target_beta_val) 
    elif beta_mode == 'linear':
        beta_fn = linear_beta_scheduler(max_epochs=num_epochs, target_val = target_beta_val) 

    alltrain = []
    allval = []
    for epoch in range(num_epochs):
        print('epoch', epoch)
        train_metrics = run_train_epoch(model, loss_mode, train_loader, optimizer, beta_fn(epoch))
        val_metrics = get_val_metrics(model, loss_mode, val_loader)
        print(epoch, train_metrics)
        print(epoch, val_metrics)

        # alltrain = {}
        # allval = {}
        # for metricss in [train_metrics, val_metrics]
        # for k,v in train_metrics:
        #     if k not in alltrain_keys


        alltrain.append(train_metrics)
        allval.append(val_metrics)

        #TODO : add plotting code for metrics (required for multiple parts)
        plt.close()
        # i = 0
        cmap = { 0:'k',1:'b',2:'y',3:'g',4:'r' }
        legend = list(range(2 * len(train_metrics.keys())))
        legenda = list(range(2 * len(train_metrics.keys())))
        # ipdb.set_trace()
        # alltrain is a list of dictionaries, each dictionary has 1 or 2 keys
        for metricss_i, metricss in enumerate([alltrain, allval]):
            # each dictionary has 1 or 2 keys
            for j, metrics in enumerate(metricss):
                i = (0 if metricss_i == 0 else len(train_metrics.keys()))
                for k,v in metrics.items():
                    print("plotting!",epoch,k,v)
                    legend[i] = k + " " + ("train" if i < len(train_metrics.keys()) else "val")
                    # plt.scatter(j, v, c=cmap[i])
                    legenda[i] = plt.scatter(j, v, 
                    label = legend[i], 
                    color = cmap[i])
                    i += 1

        plt.yscale('log')
        plt.legend(handles = legenda, labels=legend)
        plt.savefig(log_dir+"losses.jpeg")
        plt.close()

        if True: 
        # if (epoch+1)%eval_interval == 0:
            # print(epoch, train_metrics)
            # print(epoch, val_metrics)

            vis_recons(model, vis_x, 'data/'+log_dir+ '/epoch_'+str(epoch))
            if loss_mode == 'vae':
                vis_samples(model, 'data/'+log_dir+ '/epoch_'+str(epoch) )


if __name__ == '__main__':
    #TODO: Experiments to run : 
    #2.1 - Auto-Encoder
    # Run for latent_sizes 16, 128 and 1024
    # main('ae_latent1024', loss_mode = 'ae',  num_epochs = 20, latent_size = 1024)
    main('ae_latent16', loss_mode = 'ae',  num_epochs = 20, latent_size = 16)
    main('ae_latent128', loss_mode = 'ae',  num_epochs = 20, latent_size = 128)

    #Q 2.2 - Variational Auto-Encoder
    # main('vae_latent1024', loss_mode = 'vae', num_epochs = 20, latent_size = 1024)

    #Q 2.3.1 - Beta-VAE (constant beta)
    #Run for beta values 0.8, 1.2
    # main('vae_latent1024_beta_constant0.8', loss_mode = 'vae', beta_mode = 'constant', target_beta_val = 0.8, num_epochs = 20, latent_size = 1024)
    # main('vae_latent1024_beta_constant1.2', loss_mode = 'vae', beta_mode = 'constant', target_beta_val = 1.2, num_epochs = 20, latent_size = 1024)

    #Q 2.3.2 - VAE with annealed beta (linear schedule)
    # main('vae_latent1024_beta_linear1', loss_mode = 'vae', beta_mode = 'linear', target_beta_val = 1, num_epochs = 20, latent_size = 1024)