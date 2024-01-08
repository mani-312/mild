import torch
import numpy as np
import tqdm
import math
from datasets import get_dataset, data_transform, inverse_data_transform
import os
from torchvision.utils import make_grid, save_image
from evaluation.fid_score import get_fid, get_fid_stats_path
import shutil
import pickle


def get_sigmas(config):
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                               config.model.num_classes))).float().to(config.device)
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
        ).float().to(config.device)

    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas

def dynamicGamma(t,start):
    # Celeba
    # T = 1
    # 1 --  0.4(<200),0.5(<300),0.6(<400),0.7
    # 2 -- 0.4(<300),0.5(<400),0.6

    # T = 2
    # 1 -- 0.2(<300),0.3(<400),0.4
    # 2 -- 0.2(<300),0.25(<400),0.3

    # T = 3(500*3)
    # 1 -- 0.1(<500),0.15(<750),0.2(<1000),0.25

    # Cifar 10
    # 232*5
    # 1 -- 0.1(<250),0.15(<500),0.2(<750),0.25(<1k),0.3
    if t<500:
        return 0.1
    elif t<600:
        return 0.12
    elif t<800:
        return 0.15
    elif t<1000:
        return 0.17
    elif t<1250:
        return 0.2
    return 0.25

    # p = 1/(1-start)
    # a = 1-(p**(-1-math.log2(t//250 + 1)))
    # return min(a,0.9)

@torch.no_grad()
def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,gamma = 0.0,
                             gamma_dynamic = False,
                             final_only=False, verbose=False, denoise=True):
    images = []
    time_step = 1

    with torch.no_grad():
        print("---------------- Anneal Langevin Dynamics -------------")
        print("---------------- L = {} --------".format(sigmas.shape))
        print("---------------- T = {} --------".format(n_steps_each))
        print("---------------- Gamma = {} --------".format(gamma))
        print("---------------- Gamma Dynamic = {}".format(gamma_dynamic))

        V = torch.zeros_like(x_mod)

        for c, sigma in enumerate(sigmas):
            if c%50 == 0:
                print("Sigma == ",sigma)
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                # grad = scorenet(x_mod, labels)
                grad = scorenet(x_mod + gamma*V, labels)

                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()

                if gamma_dynamic:
                    gamma = dynamicGamma(time_step,0.4)
                
                V = gamma*V + step_size * grad
                
                x_mod = x_mod + V + noise * np.sqrt(step_size * 2)

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2
                time_step = time_step+1

                if not final_only:
                    images.append(x_mod.to('cpu'))
            if verbose:
                print("level: {}, gamma: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        c, gamma,step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images


@torch.no_grad()
def anneal_Langevin_dynamics_graph(config, args,scorenet, sigmas, n_steps_each=200, step_lr=0.000008,gamma = 0.0,
                             gamma_dynamic = False, denoise=True):
    images = []
    time_step = 1
    fids = {}

    init_samples = torch.rand(config.fast_fid.num_samples, config.data.channels,
                                          config.data.image_size, config.data.image_size)
    init_samples = data_transform(config, init_samples)
    V = torch.zeros_like(init_samples)

    num_iters = config.fast_fid.num_samples // config.fast_fid.batch_size
    output_path = os.path.join(args.image_folder, 'T_{}'.format(n_steps_each))
    os.makedirs(output_path, exist_ok=True)
    batch_size = config.fast_fid.batch_size
    

    
    with torch.no_grad():
        print("---------------  Anneal Langevin Dynamics -------------")
        print("---------------- L = {} --------".format(sigmas.shape))
        print("---------------- T = {} --------".format(n_steps_each))
        print("---------------- Gamma = {} --------".format(gamma))
        print("----------------- Gamma Dynamic = {}".format(gamma_dynamic))


        for c, sigma in enumerate(sigmas):
            labels = torch.ones(batch_size, device=config.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            for s in range(n_steps_each):
                print("Time_step == ",time_step)

                for i in range(num_iters):
                    x_mod = init_samples[i*batch_size : (i+1)*batch_size,:,:,:]
                    V_mod = V[i*batch_size : (i+1)*batch_size,:,:,:]

                    x_mod = x_mod.to(config.device)
                    V_mod = V_mod.to(config.device)

                    # grad = scorenet(x_mod, labels)
                    grad = scorenet(x_mod + gamma*V_mod, labels)

                    noise = torch.randn_like(x_mod)
                    V_mod = gamma*V_mod + step_size * grad
                
                    
                    x_mod = x_mod + V_mod + noise * np.sqrt(step_size * 2)

                    init_samples[i*batch_size : (i+1)*batch_size,:,:,:] = x_mod.cpu()
                    V[i*batch_size : (i+1)*batch_size,:,:,:] = V_mod.cpu()
                    del x_mod
                    del V_mod
                if time_step>20 and time_step<(n_steps_each*500-10) and time_step%5!=0:
                    time_step = time_step+1
                    continue

                os.makedirs(output_path, exist_ok=True)
                final_samples = init_samples
                for id, sample in enumerate(final_samples):
                    sample = sample.view(config.data.channels,
                                         config.data.image_size,
                                         config.data.image_size)

                    sample = inverse_data_transform(config, sample)

                    save_image(sample, os.path.join(output_path, 'sample_{}.png'.format(id)))

                stat_path = get_fid_stats_path(args, config, download=True)
                fid = get_fid(stat_path, output_path)
                fids[time_step] = fid
                with open(os.path.join(args.image_folder, 'T_{}_fids.pickle'.format(n_steps_each)), 'wb') as handle:
                    pickle.dump(fids, handle, protocol=pickle.HIGHEST_PROTOCOL)

                print("Gamma: {}, fid: {}".format(gamma, fid))
                shutil.rmtree(output_path)

                
                time_step = time_step+1


        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(batch_size, device=config.device)
            last_noise = last_noise.long()
            # x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            # images.append(x_mod.to('cpu'))


            for i in range(num_iters):
                x_mod = init_samples[i*batch_size : (i+1)*batch_size,:,:,:]
                V_mod = V[i*batch_size : (i+1)*batch_size,:,:,:]

                x_mod = x_mod.to(config.device)
                V_mod = V_mod.to(config.device)

                # original
                # x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)

                # Nestrov
                grad = scorenet(x_mod + gamma*V_mod, last_noise)
                V_mod = gamma*V_mod + sigmas[-1] ** 2 * grad
                x_mod = x_mod + V_mod

                init_samples[i*batch_size : (i+1)*batch_size,:,:,:] = x_mod.cpu()
                V[i*batch_size : (i+1)*batch_size,:,:,:] = V_mod.cpu()
                del x_mod
                del V_mod

            os.makedirs(output_path, exist_ok=True)
            final_samples = init_samples
            for id, sample in enumerate(final_samples):
                sample = sample.view(config.data.channels,
                                         config.data.image_size,
                                         config.data.image_size)

                sample = inverse_data_transform(config, sample)

                save_image(sample, os.path.join(output_path, 'sample_{}.png'.format(id)))

            stat_path = get_fid_stats_path(args, config, download=True)
            fid = get_fid(stat_path, output_path)
            fids[time_step] = fid
            with open(os.path.join(args.image_folder, 'T_{}_fids.pickle'.format(n_steps_each)), 'wb') as handle:
               pickle.dump(fids, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print("Gamma: {}, fid: {}".format(gamma, fid))
            shutil.rmtree(output_path)

        


@torch.no_grad()
def anneal_Langevin_dynamics_inpainting(x_mod, refer_image, scorenet, sigmas, image_size,
                                        n_steps_each=100, step_lr=0.000008):
    """
    Currently only good for 32x32 images. Assuming the right half is missing.
    """

    images = []

    refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
    refer_image = refer_image.contiguous().view(-1, 3, image_size, image_size)
    x_mod = x_mod.view(-1, 3, image_size, image_size)
    cols = image_size // 2
    half_refer_image = refer_image[..., :cols]
    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            for s in range(n_steps_each):
                images.append(x_mod.to('cpu'))
                corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                x_mod[:, :, :, :cols] = corrupted_half_image
                noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_size * grad + noise
                print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                                                                         grad.abs().max()))

        return images

@torch.no_grad()
def anneal_Langevin_dynamics_interpolation(x_mod, scorenet, sigmas, n_interpolations, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False):
    images = []

    n_rows = x_mod.shape[0]

    x_mod = x_mod[:, None, ...].repeat(1, n_interpolations, 1, 1, 1)
    x_mod = x_mod.reshape(-1, *x_mod.shape[2:])

    for c, sigma in enumerate(sigmas):
        
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            grad = scorenet(x_mod, labels)

            noise_p = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            noise_q = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            angles = torch.linspace(0, np.pi / 2., n_interpolations, device=x_mod.device)

            noise = noise_p[:, None, ...] * torch.cos(angles)[None, :, None, None, None] + \
                        noise_q[:, None, ...] * torch.sin(angles)[None, :, None, None, None]

            noise = noise.reshape(-1, *noise.shape[2:])
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()

            x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print(
                    "level: {}, step_size: {}, image_norm: {}, grad_norm: {}, snr: {}".format(
                        c, step_size, image_norm.item(), grad_norm.item(), snr.item()))


    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images