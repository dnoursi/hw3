from email import generator
from glob import glob
import os
import torch
from utils import get_fid, interpolate_latent_space, save_plot
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from torchvision.datasets import VisionDataset
from tqdm import tqdm
import ipdb

def build_transforms():
    # TODO 1.2: Add two transforms:
    # 1. Convert input image to tensor.
    # 2. Rescale input image to be between -1 and 1.
    # return transforms.Compose([])
    ds_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(.5,.5)])
    return ds_transforms


def get_optimizers_and_schedulers(gen, disc):
    # TODO 1.2 Get optimizers and learning rate schedulers.
    # 1. Construct the optimizers for the discriminator and generator.
    # Both should use the Adam optimizer with learning rate of .0002 and Beta1 = 0, Beta2 = 0.9.
    # 2. Construct the learning rate schedulers for the generator and discriminator.
    # The learning rate for the discriminator should be decayed to 0 over 500K steps.
    # The learning rate for the generator should be decayed to 0 over 100K steps.

    optim_discriminator  = torch.optim.Adam(disc.parameters(), betas=[0., 0.9], lr=0.0002)
    scheduler_discriminator = torch.optim.lr_scheduler.StepLR(
        optim_discriminator, step_size = 5e2, gamma = 0.99) #, last_epoch = 10) #, initial_lr=0.0002)
    optim_generator  = torch.optim.Adam(gen.parameters(), betas=[0., 0.9], lr=0.0002)
    scheduler_generator = torch.optim.lr_scheduler.StepLR(
        optim_generator, step_size = 1e2, gamma = 0.99) #, last_epoch = 10)
        # LinearLR(discriminator,1,0, 1e5)
        # LinearLR(generator,1,0, 5e5)
    return (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    )


class Dataset(VisionDataset):
    def __init__(self, root, transform=None):
        super(Dataset, self).__init__(root)
        self.file_names = glob(os.path.join(self.root, "*.jpg"), recursive=True)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_names[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.file_names)


def train_model(
    gen,
    disc,
    num_iterations,
    batch_size,
    lamb=10,
    prefix=None,
    gen_loss_fn=None,
    disc_loss_fn=None,
    log_period=10000,
):
    torch.backends.cudnn.benchmark = True
    ds_transforms = build_transforms()
    train_loader = torch.utils.data.DataLoader(
        Dataset(root="datasets/CUB_200_2011_32", transform=ds_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    ) = get_optimizers_and_schedulers(gen, disc)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    # scaler = torch.cuda.amp.GradScaler()

    iters = 0
    fids_list = []
    iters_list = []
    while iters < num_iterations:
        for train_batch in tqdm(train_loader):
            # print("looping, trainbatchshape is", train_batch.shape, "iters is", iters)
            # with torch.cuda.amp.autocast(enabled=False):
            with torch.cuda.amp.autocast():
                train_batch = train_batch.to(device = ("cuda" if torch.cuda.is_available() else "cpu"))
                # TODO 1.2: compute generator outputs and discriminator outputs
                # 1. Compute generator output -> the number of samples must match the batch size.
                # 2. Compute discriminator output on the train batch.
                # 3. Compute the discriminator output on the generated data.
                # print(batch_size)
                fake_data = gen.forward(train_batch.shape[0]) # batch_size)
                # fake_data = fake_data.detach().to(device = ("cuda" if torch.cuda.is_available() else "cpu"))
                discrim_real = disc.forward(train_batch)
                discrim_fake = disc.forward(fake_data.detach())

                # TODO: 1.5 Compute the interpolated batch and run the discriminator on it.
                # To compute interpolated data, draw eps ~ Uniform(0, 1)
                # interpolated data = eps * fake_data + (1-eps) * real_data
                if fake_data.shape != train_batch.shape:
                    ipdb.set_trace()
                eps = torch.randn((1,)).item()#.to(device = ("cuda" if torch.cuda.is_available() else "cpu")) # torch.normal(0.,1.,(1,))
                # interp = eps * fake_data.detach() + (1-eps) * train_batch
                interp = eps * fake_data + (1-eps) * train_batch
                discrim_interp = disc.forward(interp)

                discriminator_loss = disc_loss_fn(
                    discrim_real, discrim_fake, discrim_interp, interp, lamb
                )
            optim_discriminator.zero_grad(set_to_none=True)
            scaler.scale(discriminator_loss).backward()
            scaler.step(optim_discriminator)
            scheduler_discriminator.step()

            if iters % 5 == 0:
                # with torch.cuda.amp.autocast(enabled=False):
                with torch.cuda.amp.autocast():
                    # TODO 1.2: Compute samples and evaluate under discriminator.
                    # fake_data = gen.forward(train_batch.shape[0])
                    # fake_data = gen.forward()
                    discrim_fake = disc.forward(fake_data)
                    generator_loss = gen_loss_fn(discrim_fake)
                optim_generator.zero_grad(set_to_none=True)
                scaler.scale(generator_loss).backward()
                scaler.step(optim_generator)
                scheduler_generator.step()

            if (iters % 150 == 0) and iters != 0:
                with torch.no_grad():
                    # with torch.cuda.amp.autocast(enabled=False):
                    with torch.cuda.amp.autocast():
                        # TODO 1.2: Generate samples using the generator, make sure they lie in the range [0, 1].

                        generated_samples = gen.forward(100)  # train_batch.shape[0] # batch_size
                        generated_samples = generated_samples.detach()
                        generated_samples = (generated_samples + 1) / 2.

                    save_image(
                        generated_samples.data.float(),
                        prefix + "samples_{}.png".format(iters),
                        nrow=10,
                    )
                    save_plot(
                        iters_list,
                        fids_list,
                        xlabel="Iterations",
                        ylabel="FID",
                        title="FID vs Iterations",
                        filename=prefix + "fid_vs_iterations",
                    )
                    if iters % log_period == 0:
                    # torch.jit.save(gen, prefix + "/generator.pt")
                        # torch.jit.save(disc, prefix + "/discriminator.pt")
                        torch.save(gen, prefix + "/generator.pt")
                        torch.save(disc, prefix + "/discriminator.pt")
                        fid = fid = get_fid(
                            gen,
                            dataset_name="cub",
                            dataset_resolution=32,
                            z_dimension=128,
                            batch_size=batch_size,
                            num_gen=10_000,
                        )
                        print(f"Iteration {iters} FID: {fid}")
                        print("Num total iters is", num_iterations)
                        fids_list.append(fid)
                        iters_list.append(iters)

                        interpolate_latent_space(
                            gen, prefix + "interpolations_{}.png".format(iters)
                        )
            scaler.update()
            iters += 1
            # del train_batch
            # torch.cuda.empty_cache()
            
    fid = get_fid(
        gen,
        dataset_name="cub",
        dataset_resolution=32,
        z_dimension=128,
        batch_size=batch_size,
        num_gen=50_000,
    )
    print(f"Final FID (Full 50K): {fid}")
