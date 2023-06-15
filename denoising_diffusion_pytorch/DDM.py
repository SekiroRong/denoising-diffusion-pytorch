# import torch
# import torch.nn as nn
# from denoising_diffusion_pytorch import Unet, GaussianDiffusion

# class dummy_DDM(nn.Module):
#     def __init__(self, ):
#         super().__init__()
#         self.dummy_ddm()
    
#     def dummy_ddm(self, ):
#         self.model = Unet(
#             dim = 64,
#             dim_mults = (1, 2, 4, 8)
#         )

#         self.diffusion = GaussianDiffusion(
#             self.model,
#             image_size = 64,
#             timesteps = 1000    # number of steps
#         )

#         self.training_images = torch.rand(8, 3, 64, 64) # images are normalized from 0 to 1
#         loss = self.diffusion(self.training_images)
#         loss.backward()


#     def patch2grad(self, patch):
#         # patch_rgb = patch[:, :, :3]
#         # patch_depth = patch[:, :, 3]
#     # predict the noise residual with unet, NO grad!
#         with torch.no_grad():
#             # add noise
#             epsilon_theta = self.model(patch,  torch.full((8,), 1, dtype = torch.long))

#             # perform guidance (high scale from paper!)
#             return epsilon_theta

# if __name__ == "__main__":
#     dummy_ddm = dummy_DDM()
#     dummy_images = torch.rand(8, 3, 64, 64)
#     print(dummy_ddm.patch2grad(dummy_images).shape)

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    'path/to/your/images',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()