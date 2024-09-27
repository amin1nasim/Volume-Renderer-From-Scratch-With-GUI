import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import pickle
from utilities import *
from import_export_slicer import * 
import nibabel as nib
import argparse
import sys
import time
import os.path
sys.path.append('..')
from modules.render import *

# Manual seed for reproducibility
manualSeed = 999
torch.manual_seed(manualSeed)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Cuda is available' if torch.cuda.is_available() else 'Cuda is not available')

# Parsing arguments
parser = argparse.ArgumentParser(description="Interactive module for testing our volumetric renderer.")
parser.add_argument("volume", help="Path to the niff file to be rendered")
parser.add_argument("-s", "--size", type=int, default=128,
           help="The resolution (number of pixels: size ^ 2) of the final image")
parser.add_argument("-f", "--focal_coef", type=float, default=1.,
           help="Coefficient for focal lenght. Increase --focal_coef same proportion as --size to get same camera behavior")
parser.add_argument("-t", "--transfer_function", default='../tfs/CT-Bones.vp',
           help="File path to .vp file containing arbitrary piece-wise transfer fuction")
args = parser.parse_args()

# Camera properties
focal = 138.88887889922103
image_size = args.size
focal_coe = 2. * args.focal_coef
radius = 4.0311
t_near = radius - np.sqrt(3)/2
t_far = radius + np.sqrt(3)/2

# Load .nii file
rv, rv_min, rv_max =  volume_loader(args.volume, normalize_scalars=True, with_grad=False)

print(f"Volume Shape: N:{rv.shape[0]} C:{rv.shape[1]} D:{rv.shape[2]} H:{rv.shape[3]} W:{rv.shape[4]}")
print(f"Volume Min:  {rv_min}\tVolumeMax:  {rv_max}")

# Loading transfer function (TF)
tf, _ = read_slicer_tf(args.transfer_function)

# Normalizing the TF same as volume
tf[:, 0] = (tf[:, 0] - rv_min) / (rv_max - rv_min)


# Initializing Volume Renderer
volume_renderer = VolumeRenderer4(
    image_size, image_size, focal, focal_coe,
    t_near, t_far, tf_mode='truth', volume=rv,
    gt_plinear_tf=tf, t_delta=0.01)

volume_renderer.to(device)
vr_attr = volume_renderer

fig = visualize_transfer_func(vr_attr._ground_truth_tf, domain=(-0.1, 1.), device=device)

class interactive_tf:
    def __init__(self, volume_renderer, device):
        # Initialization of tf and vr
        self.vr = volume_renderer
        self.tf = self.vr.gt_plinear_tf.clone().cpu().numpy()
        self.vr.to(device)

        # Camera variables
        self.theta = 0.
        self.phi = 90.
        self.radius = 4.
        self.target = torch.tensor([0., 0., 0.], device=device)
        self.peek_pose = specific_look_at_sphere(
            self.theta, self.phi, self.radius, self.target, None)

        # Interactive variables
        self.selected_idx_plt = 0 # idx of inital selected point
        self.ignore = False # ignore events

    def __del__(self):
        print('Destructor called, Employee deleted.')

    def render_image(self, image):
        return image.detach().to('cpu').permute(1,2,0)

    def interact(self):
        ##### Setup window #####
        # Orientation
        left_margin = 0.025
        right_margin = 0.995
        top_margin = 0.85
        # Create a window
        fig, (ax_tf, ax_img) = plt.subplots(1, 2, figsize=(9.5, 6), squeeze=True)
        plt.subplots_adjust(left=left_margin, top=top_margin, right=right_margin, wspace = 0.02)


        ##### On the transferfunction side #####

        # The horizontal axie
        num_points = len(self.tf)
        x = self.tf[:, 0]

        # Plot points with color
        # Initializing with the assumption that the first point is selected
        coll = ax_tf.scatter(x, self.tf[:, -1], color=self.tf[:, 1:-1], s=200, linewidths=0.8, picker=1,
                      edgecolors=[(1,0,1,1)]+['black']*len(x-1), zorder=11)

        # Plot the line interpolating btw points
        l, = ax_tf.plot(x, self.tf[:, -1], color=(0,0,0,0.5), lw=2, zorder=10)
        left_l  = ax_tf.axhline(y=self.tf[0, -1], xmin=-0.5, xmax=self.tf[0, 0], color=(0,0,0,0.5), lw=2, zorder=10)
        right_l = ax_tf.axhline(y=self.tf[-1, -1], xmin=self.tf[-1, 0], xmax=1.5, color=(0,0,0,0.5), lw=2, zorder=10)
        # Plot grid
        ax_tf.grid(True)
        ax_tf.margins(x=0)
        ax_tf.axes.set_xlim(-0.05, 1.05)
        ax_tf.axes.set_ylim(-0.05, 1.05)

        # Position of sliders
        slider_length = 0.40
        R = plt.axes([left_margin, top_margin + 0.12, slider_length, 0.02])
        G = plt.axes([left_margin, top_margin + 0.09, slider_length, 0.02])
        B = plt.axes([left_margin, top_margin + 0.06, slider_length, 0.02])
        A = plt.axes([left_margin, top_margin + 0.03, slider_length, 0.02])
        THETA = plt.axes([0.53, top_margin + 0.12, slider_length, 0.02])
        PHI = plt.axes([0.53, top_margin + 0.09, slider_length, 0.02])
        RADIUS = plt.axes([0.53, top_margin + 0.06, slider_length, 0.02])
        SAMPLE_GAP = plt.axes([0.53, top_margin + 0.03, slider_length, 0.02])
        SAVE_BUTTON = plt.axes([0.88, 0.02, 0.1, 0.05])

        # Initialization of silders
        delta_f = 0.05 # delta for slider movements
        sr = Slider(R, 'R', 0., 1.0, valinit=self.tf[0][1], valstep=delta_f)
        sg = Slider(G, 'G', 0., 1.0, valinit=self.tf[0][2], valstep=delta_f)
        sb = Slider(B, 'B', 0., 1.0, valinit=self.tf[0][3], valstep=delta_f)
        sa = Slider(A, 'A', 0., 1.0, valinit=self.tf[0][4], valstep=delta_f)
        st = Slider(THETA, 'T', 0, 360, valinit=self.theta, valstep=1)
        sp = Slider(PHI, 'P', 0, 180, valinit=self.phi, valstep=1)
        sradius = Slider(RADIUS, 'Ra', 1., 10., valinit=self.radius, valstep=0.5)
        ssample_gap = Slider(SAMPLE_GAP, 'gap', 0.001, 0.01, valinit=self.vr.t_delta, valstep=0.001)
        # Initialization of buttons
        bsave_button = Button(SAVE_BUTTON, 'Save')
        ##### On the image side ####
        # Get rid of the axies
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # Show iamge
        img = self.vr(self.peek_pose)
        ax_img.imshow(self.render_image(img[0]))
        # On click event
        def on_pick(event):
            if event.ind[0] != self.selected_idx_plt:
                # Select the new point
                coll._edgecolors[self.selected_idx_plt] = (0, 0, 0, 1)
                self.selected_idx_plt = event.ind[0]
                coll._edgecolors[self.selected_idx_plt,:] = (1, 0, 1, 1)

                # Update the sliders for the new point
                self.ignore = True
                sr.set_val(self.tf[self.selected_idx_plt][1])
                sg.set_val(self.tf[self.selected_idx_plt][2])
                sb.set_val(self.tf[self.selected_idx_plt][3])
                sa.set_val(self.tf[self.selected_idx_plt][4])
                self.ignore = False
                fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', on_pick)

        # RGBA sliders manipulation event
        def update(val):
            if not self.ignore:
                # Update TF matrix
                self.tf[self.selected_idx_plt] =\
                np.array((self.tf[self.selected_idx_plt][0], sr.val, sg.val, sb.val, sa.val))

                # Update UI color
                coll._facecolors[self.selected_idx_plt, :-1] =\
                self.tf[self.selected_idx_plt, 1:-1]

                # Update point location
                coll._offsets[self.selected_idx_plt][1] =\
                self.tf[self.selected_idx_plt, -1]

                # Update the line
                l.set_ydata(self.tf[:, -1])
                if self.selected_idx_plt == num_points - 1:
                    right_l.set_ydata(self.tf[-1, -1])
                elif self.selected_idx_plt == 0:
                    left_l.set_ydata(self.tf[0, -1])
                # Update image side
                self.vr.gt_plinear_tf = torch.from_numpy(self.tf)
                self.vr.to(device)
                img = self.vr(self.peek_pose)
                ax_img.clear()
                ax_img.set_xticks([])
                ax_img.set_yticks([])
                ax_img.imshow(self.render_image(img[0]))
                fig.canvas.draw_idle()


        # Connecting silders to corresponding event
        sr.on_changed(update)
        sg.on_changed(update)
        sb.on_changed(update)
        sa.on_changed(update)

        # Connecting buttons to events
        def save_tf(event):
            temp = self.tf.copy()
            temp[:, 0] = temp[:, 0] * (rv_max - rv_min).item() + rv_min.item()
            np.savetxt('transfer_funcs/' + str(int(time.time())) + '.csv',
                        temp, delimiter=',')


        bsave_button.on_clicked(save_tf)

        # Camera Sliders manipulation event
        def camera_update(val):
            self.theta = st.val
            self.phi = sp.val
            if self.radius != sradius.val:
                self.radius = sradius.val
                self.vr.n = self.radius - 0.866025403784438
                self.vr.f = self.radius + 0.866025403784438
            self.vr.t_delta = ssample_gap.val

            self.peek_pose = specific_look_at_sphere(
                self.theta, self.phi, self.radius, self.target, up=None)
            img = self.vr(self.peek_pose)
            ax_img.clear()
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.imshow(self.render_image(img[0]))

        st.on_changed(camera_update)
        sp.on_changed(camera_update)
        sradius.on_changed(camera_update)
        ssample_gap.on_changed(camera_update)
        # Closing the window event
        def on_close(event):
            plt.close('all')
            plt.pause(1)

        fig.canvas.mpl_connect('close_event', on_close)
        plt.show(block=True)

itf = interactive_tf(vr_attr, device=device).interact()
