import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from modules.embedder import Embedder
import numpy as np
import inspect
from contextlib import contextmanager

# GENERATION 2
class TF_for_VR2(nn.Module):
    def __init__(self, embedf=6, layers=4, width=64, dim_in=1, manualSeed=999):

        assert layers > 1, "Number of layers should be greater than 0"
        assert dim_in == 1 or dim_in == 2,\
            f"Input features should be either 1, or 2. {self.dim_in} is invalid for dim_in."
        self.__dim_in = dim_in
        torch.manual_seed(manualSeed)
        super().__init__()
        
        # Embedder
        self.embedder = Embedder(embedf, embedf-1, input_dims=1)

        # Transfer Function
        fc_layers = []
        fc_layers += [nn.Linear(self.embedder.output_dim(), width), nn.LayerNorm(width), nn.ReLU(inplace=True)]
        fc_layers += [l for _ in range(layers-2) for l in [nn.Linear(width, width), nn.LayerNorm(width),
                                            nn.ReLU(inplace=True)]]
        fc_layers += [nn.Linear(width, 4)]
        fc_layers += [nn.Sigmoid()]
        self.main = nn.Sequential(*fc_layers)

        if self.__dim_in == 2:
            fc_layers = []
            fc_layers += [nn.Linear(self.embedder.output_dim(), width), nn.LayerNorm(width), nn.ReLU(inplace=True)]
            fc_layers += [l for _ in range(layers-2) for l in [nn.Linear(width, width), nn.LayerNorm(width),
                                                nn.ReLU(inplace=True)]]
            fc_layers += [nn.Linear(width, 1)]
            fc_layers += [nn.Sigmoid()]
            self.grad_opc = nn.Sequential(*fc_layers)


    def forward(self, x_input, _=None):
        assert inspect.stack()[2].function in\
        ['adaptive_tf_sampler', 'tf_sampler', 'visualize_transfer_func'] or  self.__dim_in == x_input.shape[1]\
        , f"Expected input with {self.__dim_in} feature(s). {x_input.shape[1]} was given"
        
        x = x_input[:, [0]]
        embedded = self.embedder(x)
        output = self.main(embedded)

        if x_input.shape[1] == 1:
            return output

        else:
            x = x_input[:, [1]]
            embedded = self.embedder(x)
            grad_out = self.grad_opc(embedded)
            final_opacity = output[:, [-1]] * grad_out
            return torch.cat((output[:, :-1], final_opacity), dim=1)

class VolumeRenderer4(nn.Module):
    def __init__(self, H, W, focal, focal_coe, n, f, t_delta=0.01, dmod=1., dt_unit_coe=50.,
     tf_mode=None, volume=None, gt_plinear_tf=None, use_nerf_compositing=False):
        super().__init__()
        self.H = H
        self.W = W
        self.n = n
        self.f = f
        self.t_delta = t_delta
        self.volume_idx = 0
        self.tf_mode = tf_mode
        self.focal = focal * focal_coe
        self.nerf = use_nerf_compositing
        self.register_buffer('volume', volume, persistent=False)
        self.register_buffer('gt_plinear_tf', gt_plinear_tf, persistent=True)
        self.register_buffer('gt_opacity_gradient', torch.tensor([[0., 1.], [1., 1.]]), persistent=True)
        self.register_buffer('dmod', torch.tensor([dmod,]), persistent=True)
        self.register_buffer('dt_unit_coe', torch.tensor([dt_unit_coe,]), persistent=True)
        self.register_buffer('bg', torch.tensor([0., 0., 0.]), persistent=True)
        self.custom_architecture = TF_for_VR2()
        self.custom_architecture_2 = None
    @contextmanager
    def brighten(self, new_coe=100):
        temp = self.dt_unit_coe
        try:
            self.dt_unit_coe = torch.tensor([new_coe,], device=self.dt_unit_coe.device)
            yield self
        except Exception as e:
            raise
        finally:
            self.dt_unit_coe = temp

    def reset_learnables(self):
        self.custom_architecture = None

    def compare(self, c2w):
        with torch.no_grad():
            temp_tf_mode = self.tf_mode
            # Generate real image
            self.tf_mode = 'truth'
            real_img = self.forward(c2w)

            # Generate fake image
            self.tf_mode = 'custom'
            fake_img = self.forward(c2w)

            # Set mode to what it was
            self.tf_mode = temp_tf_mode
            return real_img, fake_img

    def _ground_truth_tf(self, input_x, _):
        assert self.gt_plinear_tf is not None, "Ground truth transfer function wasn't provided."
        
        n, c = input_x.shape
        assert c in [1, 2], f"Input to GT transfer function should either have 1 or 2 channels (density or density + gradient). Input with {c} channels were given."
        output = torch.empty((n, 4), device=input_x.device)
        
        x = input_x[:, [0]]
        # if density is bigger than thedensity of last component in tf
        output[(x >= self.gt_plinear_tf[-1][0]).squeeze()] = self.gt_plinear_tf[-1][1:]
        # if density is smaller than the density of first component in tf
        output[(x < self.gt_plinear_tf[0][0]).squeeze()] = self.gt_plinear_tf[0][1:]

        for i in range(self.gt_plinear_tf.shape[0]-1):
            floor = self.gt_plinear_tf[i]
            ceil = self.gt_plinear_tf[i+1]

            bin = ((x >= floor[0]) & (x < ceil[0])).squeeze()
            bin_val = x[bin]
            bin_val = (bin_val - floor[0]) / (ceil[0] - floor[0])
            bin_val =\
             torch.lerp(floor[1:], ceil[1:], bin_val)
            output[bin] = bin_val
    
    
        # Multipling gradient opacity to density opacity for each point
        if c > 1:
            x = input_x[:, [1]]
            temp_output_alpha = output[:, -1]
            
            # If gradients are bigger than TF
            temp_gt = (x >= self.gt_opacity_gradient[-1][0]).squeeze()
            temp_output_alpha[temp_gt] *= self.gt_opacity_gradient[-1][1]
            # If gradients are smaller than TF
            temp_lr = (x < self.gt_opacity_gradient[0][0]).squeeze()
            temp_output_alpha[temp_lr] *= self.gt_opacity_gradient[0][1]
            # Linear interpolation for gradients in between
            for i in range(self.gt_opacity_gradient.shape[0]-1):
                floor = self.gt_opacity_gradient[i]
                ceil = self.gt_opacity_gradient[i+1]

                bin = ((x >= floor[0]) & (x < ceil[0])).squeeze()
                bin_val = x[bin]
                bin_val = (bin_val - floor[0]) / (ceil[0] - floor[0])
                bin_val =\
                 torch.lerp(floor[1], ceil[1], bin_val)
                temp_output_alpha[bin] *= bin_val.squeeze()
            
        return output

    def _get_rays(self, c2w):
        b_size = c2w.shape[0]
        device = c2w.device
        i, j = torch.meshgrid(torch.arange(self.W, device=device), torch.arange(self.H, device=device), indexing='xy')
        dirs = torch.stack([(i-self.W*.5)/self.focal, - (j-self.H*.5)/self.focal, -torch.ones_like(i)], -1).to(device)
        dirs = dirs.repeat([c2w.shape[0],] + [1,] * dirs.dim())
        rays_d = (dirs[..., None, :] * c2w[:, None, None,:3,:3]).sum(-1)
        rays_o = torch.broadcast_to(c2w[:, None, None,:3,-1], rays_d.shape)
        rays_d = F.normalize(rays_d, dim=3, p=2)
        mask, tmin, tmax = self._bool_intersect_box(rays_o.reshape(-1, 3), rays_d.reshape(-1, 3))
        return rays_o, rays_d, mask.view_as(rays_d[..., 0]), tmin, tmax

    def _bool_intersect_box(self, o, d):
        tminr = torch.empty_like(o[:, 0])
        tmaxr = torch.empty_like(tminr)
        total_mask = torch.zeros_like(tminr).to(torch.bool)

        for dim in [0,1,2]:
            # Find the posible two intersection with each axis
            tmin = (.5 - o[:,dim])/d[:,dim]
            tmax = (-.5 - o[:,dim])/d[:,dim]

            # Replace infty with zero
            tmin = torch.nan_to_num(tmin, nan=0., posinf=0., neginf=0.)
            tmax = torch.nan_to_num(tmax, nan=0., posinf=0., neginf=0.)

            # Sort base of t
            minreal = torch.minimum(tmin, tmax)
            maxreal = torch.maximum(tmin, tmax)

            # Replace intersections behind camera with zero
            minreal[minreal < 0.] = 0.
            maxreal[maxreal < 0.] = 0.
            
            # check it the point is on the box
            rd = minreal[:, None] * d + o
            mask = torch.norm(rd, p=torch.inf, dim=-1) < 0.50001
            tminr[mask] = minreal[mask]
            total_mask |= mask

            rd = maxreal[:, None] * d + o
            mask = torch.norm(rd, p=torch.inf, dim=-1) < 0.50001
            tmaxr[mask] = maxreal[mask]
            total_mask |= mask

        return total_mask, tminr[total_mask], tmaxr[total_mask]


    def _batch_render(self, o, d, m, tmin, tmax, n_chan=4, t_delta=0.01):
        device = o.device
        b, h, w, _ = o.shape
        assert self.tf_mode in ['custom2', 'custom', 'truth'], 'Determine transfer function mode (custom2 ,custom, truth).'
        # Reshape tensors
        o = o.reshape(-1, 3)
        d = d.reshape(-1, 3)
        m = m.reshape(-1)
        tmin = tmin.reshape(-1, 1)
        tmax = tmax.reshape(-1, 1)

        # closest and farthest samples
        min_t = self.n
        max_t = self.f

        t_list = torch.arange(min_t, max_t, t_delta, device=device)
        isInside = ((t_list >= tmin) & (t_list <= tmax)).view(-1)
        npts, nsamps = torch.sum(m) , len(t_list)
        d = d[m]
        o = o[m]

        pts = torch.kron(t_list, d).reshape(npts, nsamps, 3)
        pts = (pts + o[:, None, :]).view(-1, 3)
        
        # Check which samplers are inside the volume
        pts = pts[isInside][None, None, None]
        # we assumed the volume ranges from -0.5 to 0.5
        pts *= 2

        bg = self.bg
        assert bg.shape in [(3,), (b, 3)]\
                , f"background should have the shape of either (3,) or (b, 3). b={b} and bg.shape={bg.shape}"

        dmod = self.dmod
        if self.tf_mode == 'truth':
            T = self._ground_truth_tf
        elif self.tf_mode == 'custom':
            T = self.custom_architecture
        elif self.tf_mode == 'custom2':
            T = self.custom_architecture_2

        rv = self.volume[[self.volume_idx]]
        assert rv.shape[1] in [1, 4], 'Expected volume with either one or four channels'

        dchunk = torch.nn.functional.grid_sample(rv, pts,
                                                 align_corners=False,
                                                 mode='bilinear', padding_mode='border')
        dchunk = dchunk.reshape([dchunk.shape[1], dchunk.shape[-1]]).permute(1, 0)
        del(pts)
        if dchunk.shape[1] == 4:
            grad_magnitude = torch.norm(dchunk[:, 1:], p=2, dim = 1)
            dchunk = torch.stack([dchunk[:, 0], grad_magnitude], dim=-1)
            del(grad_magnitude)
        
        t_delta = self.dt_unit_coe * t_delta
        raw = torch.zeros([npts*nsamps, 4], device=o.device)
        raw[isInside] = T(dchunk, rv)

        if self.nerf:
            alpha = 1.-(1.-raw[:, -1].reshape(npts, nsamps))**t_delta
            cumprod_alpha = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)
            weights = alpha * cumprod_alpha[:, :-1]
            back_ground_exstinction = cumprod_alpha[:, [-1]]
            result_without_bg = torch.sum(weights[..., None] * raw[:, :-1].reshape(npts, nsamps, 3), dim=-2)
           
        else:
            sigma_dt = (raw[:, -1]).reshape(npts, nsamps, -1) * t_delta
            c_sigma_dt = (raw[:, :-1]).reshape(npts, nsamps, -1) * sigma_dt
            exp_rho_sigma_dt = torch.exp(-sigma_dt * dmod)

            com_rho_sigma_dt = torch.cumprod(exp_rho_sigma_dt, dim=-2)
            back_ground_exstinction = com_rho_sigma_dt[:, [-1], 0]

            com_rho_sigma_dt = F.pad(com_rho_sigma_dt, [0, 0, 1, 0, 0, 0], value=1.)[:,:-1]
            result_without_bg = torch.sum(c_sigma_dt * com_rho_sigma_dt, dim=-2)

        if bg.shape == (3,):
            bg = bg.reshape(1, 1, 1, 3).repeat(b, h, w, 1)
        elif bg.shape == (b, 3):
            bg = bg.reshape(b, 1, 1, 3).repeat(1, h, w, 1)

        result = bg.reshape(b*h*w, 3)
        result[m] *= back_ground_exstinction
        result[m] += result_without_bg
        result = result.view(b, h, w, 3).permute((0, 3, 1, 2))
        return result


    def _render_slice(self, o, d, m, t_delta=0.01):
        b, h, w, _ = o.shape
        assert False, "Not Implemented"

        assert self.tf_mode in ['custom', 'truth'], 'Determine transfer function mode (custom, truth).'

        min_t = self.n
        max_t = self.f

        t_list = torch.arange(min_t, max_t, t_delta, device=o.device)
        npts, nsamps = torch.sum(m) , len(t_list)
        d = d[m]
        o = o[m]

        pts = torch.kron(t_list, d).reshape(npts, nsamps, 3)
        pts = (pts + o[:, None, :])


        bg = self.bg
        dmod = self.dmod
        if self.tf_mode == 'truth':
            T = self._ground_truth_tf
        else:
            T = self.custom_architecture

        rv = self.volume[[self.volume_idx]]

        dchunk = torch.nn.functional.grid_sample(rv+1.,
                                                 2*pts[None,None],
                                                 align_corners=False,
                                                 mode='bilinear')-1.
        raw = T(dchunk.view(-1,1), rv)
        color = (raw[:, :-1])

        result = bg.repeat(b*h*w, nsamps, 1)
        result[m[..., None].repeat(1, nsamps)] = color

        result = result.view(b,h,w,nsamps,3).permute((0, 3, 4, 1, 2))
        return result

    def slice_view(self, c2w):
        o, d, m, tmin, tmax = self._get_rays(c2w)
        images = self._render_slice(o, d, m, t_delta=self.t_delta)
        return images

    def forward(self, c2w):
        o, d, m, tmin, tmax = self._get_rays(c2w)
        images = self._batch_render(o, d, m, tmin, tmax, t_delta=self.t_delta)
        return images


