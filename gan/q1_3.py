import os

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = ("cuda" if torch.cuda.is_available() else "cpu")

def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    # TODO 1.3.1: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.

    # need both values between 0 and 1, so use sigmoid, negate, mean

    # loss = F.binary_cross_entropy_with_logits(discrim_real, torch.ones_like(discrim_real)) 
    # loss += F.binary_cross_entropy_with_logits(discrim_fake, torch.zeros_like(discrim_fake))
    # loss = F.binary_cross_entropy_with_logits(discrim_real, torch.ones_like(discrim_real)) 
    # loss += F.binary_cross_entropy_with_logits(discrim_fake, torch.zeros_like(discrim_fake))
    # loss = loss.mean()
    # return - loss

    # return - nn.BCEWithLogitsLoss(real, ones) (fake,zeros)
    # return -(F.logsigmoid(discrim_real) + F.logsigmoid(1. - discrim_fake)).mean()
    return -(torch.log(discrim_real) + torch.log(1. - discrim_fake)).mean()
    # return -(discrim_real * torch.log(discrim_real) + (1. - discrim_fake) * torch.log(1. - discrim_fake)).mean()
    # pass


def compute_generator_loss(discrim_fake):
    # TODO 1.3.1: Implement GAN loss for generator.
    # return torch.log(1. - discrim_fake).mean()
    # bce_logits(real)
    loss = F.binary_cross_entropy_with_logits(discrim_fake, torch.zeros_like(discrim_fake))
    loss = loss.mean()
    return loss
    # pass


if __name__ == "__main__":
    print("q13 first print!")
    gen = Generator().to(device=("cuda" if torch.cuda.is_available() else "cpu")).to(memory_format=torch.channels_last) # 128
    # gen = Generator() #.cuda().to(memory_format=torch.channels_last) # 128
    # print(gen)
    # gen = gen.cuda()
    print(gen)
    
    disc = Discriminator().to(device=("cuda" if torch.cuda.is_available() else "cpu")).to(memory_format=torch.channels_last)
    print(disc)
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.3.2: Run this line of code.
    print("q13 second print!")
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
    )
