import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
from collections import namedtuple

from torch.utils import data
from pathlib import Path
from torch.optim import Adam, AdamW

import numpy as np
from tqdm import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

# constants
SAVE_AND_SAMPLE_EVERY = 100_000
#SAVE_AND_SAMPLE_EVERY = 1_000
UPDATE_EMA_EVERY = 10
PRINT_LOSS_EVERY = 200

# MODEL_INFO= '128-1-2-2-4-b128'
# helpers functions

# MJ -- was this causing an issue with the xyz conditioning approach?
def generate_inprint_mask(n_batch, op_num, unmask_index=None):
    """
    The mask will be True where we keep the true value and false where we want to infer the value
    So far it only supporting masking the right side of images
    """

    mask = torch.zeros((n_batch, 1, op_num), dtype=bool)
    if not unmask_index == None:
        mask[:, :, unmask_index] = True
    return mask


def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


# small helper modules

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 3, 2, 1)  # changed from kernel = 3

    
class WeightStandardizedConv2d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    
# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)


# model -- replaced Unet

class Unet1D(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # for consistency with previous save names
        self.feature_dim = dim
        self.dim_mults = dim_mults

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        
        # reshape if n_channels = 3 -- make this into a function and apply to self-conditionign
        if self.channels == 3:
            n_atoms = x.shape[-1] // 3 
            x = x.reshape((-1, n_atoms, 3))
            x = torch.transpose(x, 1, 2)

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []
        size_list = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            size_list.append(x.shape[-1]) # track sizes

            x = block2(x, t)
            x = attn(x)
            h.append(x)
            size_list.append(x.shape[-1]) # tracks sizes
            
            x = downsample(x)  

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            
            x = torch.cat((x[:, :, :size_list.pop()], h.pop()), dim = 1)
            #x = torch.cat((x, h.pop()), dim = 1)
            
            x = block1(x, t)

            x = torch.cat((x[:, :, :size_list.pop()], h.pop()), dim = 1)
            #x = torch.cat((x, h.pop()), dim = 1)
            
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        
        # reshape to 1D if n_channels = 3
        if self.channels == 3:
            x = torch.transpose(x, 1, 2)
            x = x.reshape((-1, 1, n_atoms*3))
        
        return x


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def linear_schedule(timesteps, s=0.008):
    """
    linear schedule
    """
    betas = np.linspace(0.0001, 0.02, timesteps, dtype=np.float64)
    return np.clip(betas, a_min=0, a_max=0.999)


# MJ -- add cosine scheduler


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    betas = betas.numpy()
    return np.clip(betas, a_min=0, a_max=0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        timesteps=1000,
        loss_type="l1",
        betas=None,
        beta_schedule="linear",
        unmask_number=0,
        unmask_list=[],
    ):
        super().__init__()
        self.denoise_fn = denoise_fn

        if exists(betas):
            betas = (
                betas.detach().cpu().numpy()
                if isinstance(betas, torch.Tensor)
                else betas
            )

        # which beta scheduler to use
        else:
            if beta_schedule == "linear":
                betas = linear_schedule(timesteps)
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.unmask_number = unmask_number

        # can account for exact list of masks idxs or for leading consequecutive conditons
        if len(unmask_list) > 0:
            self.unmask_index = list(unmask_list)
        elif unmask_number == 0:
            self.unmask_index = None
        else:
            self.unmask_index = [*range(unmask_number)]

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, _, l, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        denosied_x = (
            model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        )
        inprint_mask = generate_inprint_mask(b, l, self.unmask_index).to(device)

        denosied_x[inprint_mask] = x[inprint_mask]

        return denosied_x

    @torch.no_grad()
    def p_sample_loop(self, shape, samples=None, save_diffusion_progress=False):
        device = self.betas.device

        b = shape[0]
        state = torch.randn(shape, device=device)

        if not samples == None:
            assert shape == samples.shape

            inprint_mask = generate_inprint_mask(b, shape[2], self.unmask_index).to(
                device
            )
            state[inprint_mask] = samples[inprint_mask]

        # save_diffusion_progress = False
        if save_diffusion_progress:

            # MJ added to check every diffusion timestep
            state_list = []
            for i in tqdm(
                reversed(range(0, self.num_timesteps)),
                desc="sampling loop time step",
                total=self.num_timesteps,
            ):
                state = self.p_sample(
                    state, torch.full((b,), i, device=device, dtype=torch.long)
                )
                state_list.append(state.cpu().detach().numpy())
                print(np.shape(state_list))

            # save diffusion progress
            np.save(
                "./results/train_pep_AA-heavy/32-1-2-4-8-b128/AA-heavy_Unet1D_2000000s/diff_progess.npy",
                np.array(state_list),
            )

        # normal sampling procedure
        else:
            for i in tqdm(
                reversed(range(0, self.num_timesteps)),
                desc="sampling loop time step",
                total=self.num_timesteps,
            ):
                state = self.p_sample(
                    state, torch.full((b,), i, device=device, dtype=torch.long)
                )

        return state

    @torch.no_grad()
    def sample(self, op_number, batch_size=16, samples=None):
        return self.p_sample_loop((batch_size, 1, op_number), samples)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        if not self.unmask_index == None:
            b, c, l = x_start.shape
            inprint_mask = generate_inprint_mask(b, l, self.unmask_index).to(
                x_start.device
            )
            x_start[inprint_mask]
            x_noisy[inprint_mask] = x_start[inprint_mask]
        else:
            inprint_mask = None
        return x_noisy, inprint_mask

    def p_losses(self, x_start, t, noise=None):
        b, c, l = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy, inprint_mask = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.denoise_fn(x_noisy, t)

        if not inprint_mask == None:
            noise = torch.masked_select(noise, ~inprint_mask)
            x_recon = torch.masked_select(x_recon, ~inprint_mask)

        if self.loss_type == "l1":
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == "l2":
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, *_, device = *x.shape, x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)


# dataset classes


#class Dataset_traj(torch.utils.data.Dataset):
#    "Characterizes a dataset for PyTorch"
#
#    #def __init__(
#    #    self, folder, system, shared_idxs=None
#    #):  # MJ add n_conds if using when passing in conds
#    #    super().__init__()
#    #    self.folder = folder
#    #    self.shared_idxs = shared_idxs
#    #
#    #    self.data = np.load(f"{folder}/{system}_traj.npy")
#    #    self.max_data = np.max(self.data, axis=0)
#    #    self.min_data = np.min(self.data, axis=0)
#    def __init__(
#            self,
#            folder=None,
#            system=None,
#            shared_idxs=None,
#            npy_data=None,
#        ):  # MJ add n_conds if using when passing in conds
#            super().__init__()
#            self.folder = folder
#            self.shared_idxs = shared_idxs
#            # enable npys to be passed in directly
#            if npy_data is None:
#                self.data = np.load(f"{folder}/{system}_traj.npy")
#            else:
#                self.data = npy_data
#            # only normalize shared idxs during inference
#            if shared_idxs is not None:
#                self.data = self.data[:, : len(shared_idxs)]
#            self.max_data = np.max(self.data, axis=0)
#            self.min_data = np.min(self.data, axis=0)
#
#    def __len__(self):
#        "Denotes the total number of samples"
#        return np.shape(self.data)[0]
#
#    def __getitem__(self, index):
#        "Generates one sample of data"
#
#        # Select sample
#        x = self.data[index : index + 1, :]
#
#        if self.shared_idxs is not None:
#            min_c, max_c = (
#                self.min_data[self.shared_idxs],
#                self.max_data[self.shared_idxs],
#            )
#            x_c = 2 * x / (max_c - min_c)
#            x_c = x_c - 2 * min_c / (max_c - min_c) - 1
#
#            # just need to append values to match input dims
#            x_z = np.zeros((1, len(self.max_data)))
#
#            x_list = []
#            for idx in range(len(self.max_data)):
#                if idx in self.shared_idxs:
#                    c_idx = list(self.shared_idxs).index(idx)
#                    x_list.append(x_c[:, c_idx])
#                else:
#                    x_list.append(x_z[:, idx])
#
#            x = np.concatenate(x_list)[np.newaxis, ...]
#
#        # scales everything to [1, -1] for normal case
#        else:
#            x = 2 * x / (self.max_data - self.min_data)
#            x = x - 2 * self.min_data / (self.max_data - self.min_data) - 1
#
#        # x = x[ np.newaxis, ...]
#        return torch.from_numpy(x).float()

class Dataset_traj(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(
        self,
        folder=None,
        system=None,
        shared_idxs=None,
        npy_data=None,
        
    ):  # MJ add n_conds if using when passing in conds
        super().__init__()
        self.folder = folder
        self.shared_idxs = shared_idxs
        
        # for inference only
        if self.folder is None:
            self.max_data = 0
            self.min_data = 0
            
        else:
            # enable npys to be passed in directly
            if npy_data is None:
                self.data = np.load(f"{folder}/{system}_traj.npy")
            else:
                self.data = npy_data

            # only normalize shared idxs during inference
            if shared_idxs is not None:
                self.data = self.data[:, : len(shared_idxs)]

            self.max_data = np.max(self.data, axis=0)
            self.min_data = np.min(self.data, axis=0)
        
            
    def __len__(self):
        "Denotes the total number of samples"
        
        if self.folder is not None:
            return np.shape(self.data)[0]
        else:
            return 1
    
    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        x = self.data[index : index + 1, :]
        if self.shared_idxs is not None:
            
            min_c, max_c = (
                self.min_data[self.shared_idxs],
                self.max_data[self.shared_idxs],
            )
  
            x_c = 2 * x / (max_c - min_c)
            x_c = x_c - 2 * min_c / (max_c - min_c) - 1
            # just need to append values to match input dims
            x_z = np.zeros((1, len(self.max_data)))
            
            # will in missing values corresponding to conds
            x_z[0, self.shared_idxs] = x_c
            x = x_z
        # scales everything to [1, -1] for normal case
        else:
            x = 2 * x / (self.max_data - self.min_data)
            x = x - 2 * self.min_data / (self.max_data - self.min_data) - 1
            
        return torch.from_numpy(x).float()


# trainer class


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        system,
        *,
        system_for_sample=None,
        ema_decay=0.995,
        op_number=4,
        train_batch_size=32,
        sample_batch_size=None,
        train_lr=2e-5,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        fp16=False,
        save_name=None,
        step_start_ema=2000,
        scheduler_gamma=None,
        adamw=False,
        rescale=None,
    ):
        super().__init__()

        feature_dim = diffusion_model.denoise_fn.module.feature_dim
        dim_mults = diffusion_model.denoise_fn.module.dim_mults

        MODEL_INFO = f"{feature_dim }-"
        for w in dim_mults:
            MODEL_INFO += f"{w}-"
        MODEL_INFO += f"b{train_batch_size}-"
        MODEL_INFO += f"lr{train_lr:.1e}"
        if scheduler_gamma is not None:
            MODEL_INFO += f"-gamma{scheduler_gamma}-"
        if adamw:
            MODEL_INFO += f"-optAdamW-"
        if rescale is not None:
            MODEL_INFO += f"-rescale{rescale}-"
       
        self.RESULTS_FOLDER = Path(f"./trained_models/{system}/{MODEL_INFO}")  # remove /save_name/ to reduce dir depth
        self.RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)   
        
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.step_start_ema = step_start_ema
        self.scheduler_gamma = scheduler_gamma

        self.batch_size = train_batch_size
        self.op_number = op_number  # + 1  MJ is this only for the trainer?
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        
        if folder is None:
            self.ds = Dataset_traj()

        # only needed when initializing a new model for training
        else:
            
            np.save(str(self.RESULTS_FOLDER / f"shared_idxs.npy"), diffusion_model.unmask_index)
            np.save(str(self.RESULTS_FOLDER / f"op_number.npy"), op_number)
            
            self.ds = Dataset_traj(folder, system)

            if rescale is not None:
                self.ds.max_data[57:-20] = rescale
                self.ds.min_data[57:-20] = -rescale
                self.ds.max_data[-20:] = 1.0
                self.ds.min_data[-20:] = 0.0

            self.dl = cycle(
                data.DataLoader(
                    self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True
                )
            )

            self.sample_batch_size = train_batch_size
            if system_for_sample == None:
                self.dl_sample = self.dl
            else:
                self.ds_sample = Dataset_traj(folder, system_for_sample)
                if sample_batch_size == None:
                    self.sample_batch_size = train_batch_size
                self.dl_sample = cycle(
                    data.DataLoader(
                        self.ds_sample,
                        batch_size=sample_batch_size,
                        shuffle=True,
                        pin_memory=True,
                    )
                )

            self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
            if adamw:
                self.opt = AdamW(diffusion_model.parameters(), lr=train_lr)

            if self.scheduler_gamma is not None:
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, scheduler_gamma)

            self.step = 0

            assert (
                not fp16 or fp16 and APEX_AVAILABLE
            ), "Apex must be installed in order for mixed precision training to be turned on"

            self.fp16 = fp16
            if fp16:
                (self.model, self.ema_model), self.opt = amp.initialize(
                    [self.model, self.ema_model], self.opt, opt_level="O1"
                )

            self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "data_range": [self.ds.min_data, self.ds.max_data],
        }
        torch.save(data, str(self.RESULTS_FOLDER / f"model-{milestone}.pt"))

    def rescale_sample_back(self, sample):
        def scale_back(data, minimums, maximums):
            data = (data + 1) / 2.0 * (maximums - minimums)
            data += minimums
            return data

        max_data = self.ds.max_data
        min_data = self.ds.min_data

        sample = scale_back(sample, min_data, max_data)

        return sample
    
    def load_restart(self, load_path, device=torch.device("cuda")):
        model_data = torch.load(load_path, map_location=device)

        self.step = model_data["step"]
        self.model.load_state_dict(model_data["model"])
        self.ema_model.load_state_dict(model_data["ema"])
        self.ds.min_data = model_data["data_range"][0]
        self.ds.max_data = model_data["data_range"][1]
        self.dl = cycle(
            data.DataLoader(
                self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=True
            )
        )


    def load(self, milestone, device=torch.device("cuda")):
        model_data = torch.load(
            str(self.RESULTS_FOLDER / f"model-{milestone}.pt"),  # MJ added for cpu
            map_location=device,
        )

        self.step = model_data["step"]
        self.model.load_state_dict(model_data["model"])
        self.ema_model.load_state_dict(model_data["ema"])
        self.ds.min_data = model_data["data_range"][0]
        self.ds.max_data = model_data["data_range"][1]
        self.dl = cycle(
            data.DataLoader(
                self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=True
            )
        )

    def train(self):
        backwards = partial(loss_backwards, self.fp16)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()
                loss = self.model(data)
                backwards(loss / self.gradient_accumulate_every, self.opt)

            self.opt.step()
            self.opt.zero_grad()
            if self.scheduler_gamma is not None:
                self.scheduler.step()

            if self.step % UPDATE_EMA_EVERY == 0:
                self.step_ema()

            if self.step % PRINT_LOSS_EVERY == 0:
                print(f"Step {self.step}: {loss.item()}")

            if self.step != 0 and self.step % SAVE_AND_SAMPLE_EVERY == 0:
                milestone = self.step // SAVE_AND_SAMPLE_EVERY
                print(f'Saving at step {self.step}; milestone #{milestone}')
                #batches = num_to_groups(self.sample_batch_size, self.batch_size)

                #all_ops_list = list(
                #    map(
                #        lambda n: self.ema_model.sample(
                #            self.op_number,
                #            batch_size=n,
                #            samples=next(self.dl_sample).cuda()[:n, :],
                #        ),
                #        batches,
                #    )
                #)

                #all_ops = torch.cat(all_ops_list, dim=0).cpu()
                #all_ops = self.rescale_sample_back(all_ops)
                #np.save(
                #    str(self.RESULTS_FOLDER / f"sample-{milestone}"), all_ops.numpy()
                #)
                self.save(milestone)

            self.step += 1

        print("training completed")

