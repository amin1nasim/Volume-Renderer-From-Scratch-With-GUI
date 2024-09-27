import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.linalg
import math
import nibabel as nib

# This one up is [0, 1, 0] if not given
def specific_look_at_sphere(theta, phi, radius, target, up):
    device = target.device

    theta = math.radians(theta)
    phi = math.radians(phi)

    if up is None:
        up = torch.tensor([0., 0., 1.], device=device)

    view_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
    view_matrix[-1, -1] = 1.

    sphere_sample = torch.tensor([
                                math.cos(theta) * math.sin(phi),
                                math.sin(theta) * math.sin(phi),
                                math.cos(phi),
                                ], device=device)


    translation =  sphere_sample * radius
    view_matrix[:-1, 3] = translation

    z = torch.nn.functional.normalize(translation - target, dim=0)
    view_matrix[:-1, 2] = z

    x = torch.nn.functional.normalize(torch.linalg.cross(up, z), dim=0)
    view_matrix[:-1, 0] = x

    y = torch.nn.functional.normalize(torch.linalg.cross(z, x), dim=0)
    view_matrix[:-1, 1] = y

    return view_matrix.unsqueeze(0)

# This one up is random if not given
def look_at_sphere(batch_size, radius, target, up):
    device = target.device

    if up is None:
        up = torch.nn.functional.normalize(torch.randn((batch_size, 3), device=device))

    else:
        up = up.to(device)
        up = torch.broadcast_to(up, (batch_size, 3))

    view_matrix = torch.zeros(batch_size, 4, 4, dtype=torch.float32, device=device)
    view_matrix[:, -1, -1] = 1.

    sphere_sample = F.normalize(torch.randn(batch_size, 3, device=device))
    translation =  sphere_sample * radius
    view_matrix[:, :-1, 3] = translation

    z = torch.nn.functional.normalize(translation - target)
    view_matrix[:, :-1, 2] = z

    x = torch.nn.functional.normalize(torch.linalg.cross(up, z))
    view_matrix[:, :-1, 0] = x

    y = torch.nn.functional.normalize(torch.linalg.cross(z, x))
    view_matrix[:, :-1, 1] = y

    return view_matrix


def create_grid(dataset, num_samples):
    showcase = torch.stack([dataset[i][0] for i in torch.randint(len(dataset), (num_samples,))], 0)
    showcase = vutils.make_grid(showcase, padding=2)
    return showcase
    
def tensor2image(tensor):
	dim = tensor.dim()
	assert dim in [3, 4], f"Expected a tensor with dim dimensionality of 3 or 4. Recived {dim}"
	if dim == 3:
		return tensor.detach().cpu().permute(1, 2, 0).clamp(min=0., max=1.)
		
	elif dim == 4:
		if tensor.shape[0] == 1:
			return tensor[0].detach().cpu().permute(1, 2, 0).clamp(min=0., max=1.)
		else:
			return vutils.make_grid(tensor).cpu().permute(1, 2, 0).clamp(min=0., max=1.)
		


def old_visualize_transfer_func(F, pass_none=True, domain=(-1., 1.), device='cpu'):
    with torch.no_grad():
        x = torch.linspace(domain[0], domain[1], 1000, device=device).reshape(-1,1)
        if not pass_none:
            y = F(x).detach().cpu()
        else:
            y = F(x, None).detach().cpu()

        x = x.cpu()
        fig, (ax1, ax2) = plt.subplots(2, 1)

        fig.suptitle('Transfer Function')

        ax1.plot(x, y[:,-1], 'o-')
        ax1.set_ylim(0.0, 1.0)
        ax1.set_ylabel('Opacity')

        ax2.plot(x, y[:,0], '.-', color='red')
        ax2.plot(x, y[:,1], '.-', color='green')
        ax2.plot(x, y[:,2], '.-', color='blue')

        ax2.set_ylim(0.0, 1.0)
        ax2.set_xlabel('Density')
        ax2.set_ylabel('Colour')

        return fig

def tf_sampler(F, normalization_min, normalization_max, domain=(0., 1.), distribution=100,
            sample_gradien=False, grad_normalization_min=0., grad_normalization_max=1., grad_domain=(0., 1.), grad_distribution=100,
            pass_none=True, device='cpu'):

    with torch.no_grad():
        if type(distribution) == int:
            x = torch.linspace(domain[0], domain[1], distribution, device=device).reshape(-1,1)
        elif type(distribution) == torch.Tensor:
            x = distribution.reshape(-1, 1).to(device)
        else:
            raise Exception("Parameter distribution should be either an integer for uniform sampling or torch.Tensor") 
        if not pass_none:
            y = F(x)
        else:
            y = F(x, None)
            
        if sample_gradien:
            if type(grad_distribution) == int:
                x_grad = torch.linspace(grad_domain[0], grad_domain[1], grad_distribution, device=device).reshape(-1,1)
            elif type(grad_distribution) == torch.Tensor:
                x_grad = grad_distribution.reshape(-1, 1).to(device)

            max_opacity = x[torch.argmax(y[:,-1])][0].item()
            gradient_x = torch.cat([torch.full_like(x_grad, max_opacity, device=device), x_grad], dim=1)
            if not pass_none:
                gradient_y = F(gradient_x).detach()
            else:
                gradient_y = F(gradient_x, None).detach()
        	
            gradient_y = gradient_y[:,[-1]] / y[:,-1].max()
        	
            x_d = x * (normalization_max - normalization_min) + normalization_min
            x_g = x_grad * (grad_normalization_max - grad_normalization_min) + grad_normalization_min
            
            return (torch.cat([x_d, y], dim=1).detach(), torch.cat([x_g, gradient_y], dim=1).detach())
        	
        x = x * (normalization_max - normalization_min) + normalization_min
        return torch.cat([x, y], dim=1).detach()
    
def adaptive_tf_sampler(F, threshold=0.05, initial_number_samples=100,
                        normalization_min=0., normalization_max=1., domain=(0., 1.),
                        sample_gradien=True, grad_normalization_min=0., grad_normalization_max=1., grad_domain=(0., 1.),
                        pass_none=True, device='cpu'):
    
    with torch.no_grad():
        enough_samples = False
        x = torch.linspace(domain[0], domain[1], initial_number_samples, device=device).reshape(-1,1)
        while not enough_samples:
            if not pass_none:
                y = F(x)
            else:
                y = F(x, None)
                
            diff = torch.abs(y[1:, -1] - y[:-1, -1])
            is_far = diff > threshold
            if torch.all(is_far == False):
                enough_samples = True
            else:
                idx = torch.where(is_far)[0]
                new_values = (x[idx+1] + x[idx]) * 0.5
                x = torch.sort(torch.cat([x, new_values]), dim=0).values



        if sample_gradien:
            enough_samples = False
            x_grad = torch.linspace(grad_domain[0], grad_domain[1], initial_number_samples, device=device).reshape(-1,1)
            max_opacity = x[torch.argmax(y[:,-1])][0].item()
            
            while not enough_samples:
                gradient_x = torch.cat([torch.full_like(x_grad, max_opacity, device=device), x_grad], dim=1)
                if not pass_none:
                    gradient_y = F(gradient_x).detach()
                else:
                    gradient_y = F(gradient_x, None).detach()

                gradient_y = gradient_y[:,[-1]] / y[:,-1].max()
                diff = torch.abs(gradient_y[1:] - gradient_y[:-1])
                is_far = diff > threshold
                if torch.all(is_far == False):
                    enough_samples = True
                else:
                    idx = torch.where(is_far)[0]
                    new_values = (x_grad[idx+1] + x_grad[idx]) * 0.5
                    x_grad = torch.sort(torch.cat([x_grad, new_values]), dim=0).values
                    
            x_d = x * (normalization_max - normalization_min) + normalization_min
            x_g = x_grad * (grad_normalization_max - grad_normalization_min) + grad_normalization_min

            return (torch.cat([x_d, y], dim=1).detach(), torch.cat([x_g, gradient_y], dim=1).detach())

        x = x * (normalization_max - normalization_min) + normalization_min
        return torch.cat([x, y], dim=1).detach()


def visualize_transfer_func(F, pass_none=True, domain=(-.1, 1.), grad_domain=(0., 1.), device='cpu'):
    NUM_SAMPLES = 1000

    scaler_tf, gradient_tf = tf_sampler(F, normalization_min=0, normalization_max=1, domain=domain, distribution=NUM_SAMPLES,
            sample_gradien=True, grad_normalization_min=0., grad_normalization_max=1., grad_domain=grad_domain, grad_distribution=NUM_SAMPLES,
            pass_none=pass_none, device=device)

    
    scaler_tf = scaler_tf.cpu().numpy()
    gradient_tf = gradient_tf.cpu().numpy()

    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle('Transfer Function')

    ax1.plot(scaler_tf[:, 0], scaler_tf[:,-1], 'o-')
    ax1.set_xlabel('Density')
    ax1.set_ylabel('Opacity')
    

    ax2.plot(scaler_tf[:, 0], scaler_tf[:, 1], '.-', color='red')
    ax2.plot(scaler_tf[:, 0], scaler_tf[:, 2], '.-', color='green')
    ax2.plot(scaler_tf[:, 0], scaler_tf[:, 3], '.-', color='blue')
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel('Density')
    ax2.set_ylabel('Colour')
    
    ax3.plot(gradient_tf[:, 0], gradient_tf[:, 1], 'o-')
    ax3.set_xlabel('Gradient')
    ax3.set_ylabel('Opacity')
    fig.tight_layout()
    return fig


### Loading volumes
def volume_loader(path, with_grad=True, bad_spacing=True,
        normalize_scalars=False):
    volume = nib.load(path)
    rv = torch.from_numpy(volume.get_fdata()).float()
    rv_max = torch.max(rv)
    rv_min = torch.min(rv)

    if normalize_scalars:
        rv = (rv - rv_min) / (rv_max - rv_min)
        
    if with_grad:
        spacing =\
        [1., 1., 1.] if bad_spacing else\
          torch.from_numpy(volume.affine).diagonal()[:3].tolist()
        dx, dy, dz = torch.gradient(rv, spacing=spacing)
        gradient_mag = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2) 
        max_grad_mag = gradient_mag.max()

        # Reshape the volume
        rv = torch.stack([rv, dx, dy, dz])[None]
        rv = rv.permute(0, 1, 4, 2, 3)
        
        # Print info
        print(f"Volume Shape: N:{rv.shape[0]} C:{rv.shape[1]} D:{rv.shape[2]} H:{rv.shape[3]} W:{rv.shape[4]}")
        print(f"Volume Spacing: {spacing}")
        print(f"Maximum Gradient Magnitude: {max_grad_mag}")
        print(f"Volume Min:  {rv_min}\tVolume Max:  {rv_max}")
        print()
        
        return rv, rv_min, rv_max, max_grad_mag
    
    # Reshape the volume
    rv = rv[None, None]
    rv = rv.permute(0, 1, 4, 2, 3)
    
    # Print info
    print(f"Volume Shape: N:{rv.shape[0]} C:{rv.shape[1]} D:{rv.shape[2]} H:{rv.shape[3]} W:{rv.shape[4]}")
    print(f"Volume Min:  {rv_min}\tVolume Max:  {rv_max}")
    print()

    return rv, rv_min, rv_max
