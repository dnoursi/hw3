import torch
from cleanfid import fid
from matplotlib import pyplot as plt
import torchvision
# from torchvision.utils import save_image


def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    # TODO 1.2: Generate and save out latent space interpolations.
    # Generate 100 samples of 128-dim vectors
    # Linearly interpolate the first two dimensions between -1 and 1. 
    # Keep the rest of the z vector for the samples to be some fixed value. 
    # Forward the samples through the generator.
    # Save out an image holding all 100 samples.
    # Use torchvision.utils.save_image to save out the visualization.

    n=10
    generated = torch.randn((n*n,128,)).to(device = ("cuda" if torch.cuda.is_available() else "cpu")) 
    # torch.normal(torch.zeros(128),1., n)
    generated[:, 0] = torch.linspace(-1.,1.,n).repeat(n)
    generated[:, 1] = torch.linspace(-1.,1.,n).repeat_interleave(n)
    #     gmin = generated[:,i].min()
    #     gmax = generated[:,i].max()
    #     r = gmax - gmin
    #     generated[:, i] = (2./r)  * (generated[:, i] - gmin) - 1

    
    forwarded = gen.forward_given_samples(generated)
    forwarded = (forwarded + 1.)/2
    torchvision.utils.save_image(forwarded.data.float(), path,nrow=n)
    return forwarded
    
    # ...................

    n=100
    generated = torch.randn((n,128,)).to(device = ("cuda" if torch.cuda.is_available() else "cpu")) 
    # torch.normal(torch.zeros(128),1., n)
    for i in [0,1]:
        generated[:, i] = torch.linspace(-1.,1.,n)
    #     gmin = generated[:,i].min()
    #     gmax = generated[:,i].max()
    #     r = gmax - gmin
    #     generated[:, i] = (2./r)  * (generated[:, i] - gmin) - 1

    
    forwarded = gen.forward_given_samples(generated)
    forwarded = (forwarded + 1.)/2
    torchvision.utils.save_image(forwarded.data.float(), path,nrow=10)
    return forwarded
    
    
    pass
