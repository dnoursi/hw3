import os

import torch

from networks import Discriminator, Generator
from train import train_model
import ipdb

def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    # TODO 1.5.1: Implement WGAN-GP loss for discriminator.
    # loss = E[D(fake_data)] - E[D(real_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    # gradd = torch.autograd.grad((discrim_interp).mean(), interp, 
    gradd = torch.autograd.grad(discrim_interp.sum(), interp, 
                                create_graph=True, retain_graph=True,
                                # allow_unused=True
                                # is_grads_batched=True,
                                )[0]
    # gradd = gradd.mean(dim=0).norm()
    # ipdb.set_trace()
    gradd = gradd.norm(p=2., dim=(1,2,3))
    loss_e = (gradd - 1.) ** 2
    loss_e = loss_e.mean()
    loss = (discrim_fake - discrim_real).mean() + lamb * loss_e
    return  loss


def compute_generator_loss(discrim_fake):
    # TODO 1.5.1: Implement WGAN-GP loss for generator.
    # loss = - E[D(fake_data)]
    return - discrim_fake.mean()
    return loss


if __name__ == "__main__":
    gen = Generator().cuda().to(memory_format=torch.channels_last)
    disc = Discriminator().cuda().to(memory_format=torch.channels_last)
    prefix = "data_wgan_gp/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.5.2: Run this line of code.
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
