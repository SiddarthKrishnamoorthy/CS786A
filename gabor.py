import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from copy import deepcopy
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from skimage.color import rgb2gray
from skimage.transform import resize

PATH = "/home/ks1d/Documents/CS786A/Assignment-2/imgs/"
def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats
    print(len(powers))

def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i

# prepare filter bank kernels
kernels = []
NUM_ORIENTATIONS = 8
THETA_LIST = [0, np.pi/3, np.pi/2, 2*np.pi/3]
for theta in THETA_LIST:
    #theta = float(theta)/NUM_ORIENTATIONS  * np.pi
    for frequency in [0.125]:
        kernel = np.real(gabor_kernel(frequency, theta=theta, bandwidth=1))
        kernels.append(kernel)

def convert_to_gray(img):
    img = resize(img, (512, 512), anti_aliasing=True)
    if len(img.shape)==3:
        return rgb2gray(img)
    return img

shrink = (slice(0, None, 3), slice(0, None, 3))
#brick = img_as_float(data.load('brick.png'))[shrink]
#grass = img_as_float(data.load('grass.png'))[shrink]
#wall = img_as_float(data.load('rough-wall.png'))[shrink]
triangle = img_as_float(data.load(os.path.join(PATH, 'triangle.png')))
triangle = convert_to_gray(triangle)
triangle2 = img_as_float(data.load(os.path.join(PATH, 'triangle2.jpg')))
triangle2 = convert_to_gray(triangle2)
square = img_as_float(data.load(os.path.join(PATH, 'square1.jpg')))
square = convert_to_gray(square)
bakchodi = img_as_float(data.load(os.path.join(PATH, 'bakchod.png')))
bakchodi = convert_to_gray(bakchodi)
#image_names = ('brick', 'grass', 'wall')
image_names = ['triangle', 'triangle2', 'square', 'bakchod']
#images = (brick, grass, wall)
images = [triangle, triangle2, square, bakchodi]

# prepare reference features
ref_feats = np.zeros((1, len(kernels), 2), dtype=np.double)
#ref_feats[0, :, :] = compute_feats(brick, kernels)
ref_feats[0, :, :] = compute_feats(triangle, kernels)
#ref_feats[1, :, :] = compute_feats(grass, kernels)
#ref_feats[2, :, :] = compute_feats(wall, kernels)

print('Rotated images matched against references using Gabor filter banks:')

print('original: brick, rotated: 30deg, match result: ', end='')
feats = compute_feats(ndi.rotate(triangle, angle=190, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    print(image.shape)
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2+ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
    #return np.sqrt(ndi.convolve(image, kernel, mode='wrap'))

# Plot a selection of the filter bank kernels and their responses.
results = []
kernel_params = []
for theta in THETA_LIST:
    #theta = float(i) / NUM_ORIENTATIONS * np.pi
    for frequency in [0.125]:
        kernel = gabor_kernel(frequency, theta=theta)
        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
        kernel_params.append(params)
        # Save kernel and the power image for each image
        results.append((kernel, [power(img, kernel) for img in images]))

'''
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(5, 6))
plt.gray()

fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

axes[0][0].axis('off')

# Plot original images
for label, img, ax in zip(image_names, images, axes[0][1:]):
    ax.imshow(img)
    ax.set_title(label, fontsize=9)
    ax.axis('off')
for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
    # Plot Gabor kernel
    ax = ax_row[0]
    ax.imshow(np.real(kernel), interpolation='nearest')
    ax.set_ylabel(label, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot Gabor responses with the contrast normalized for each filter
    vmin = np.min(powers)
    vmax = np.max(powers)
    for patch, ax in zip(powers, ax_row[1:]):
        ax.imshow(patch, vmin=vmin, vmax=vmax)
        ax.axis('off')

plt.show()
'''
a = np.zeros((512,512))
b = [deepcopy(a), deepcopy(a), deepcopy(a), deepcopy(a)]
for (kernel, powers) in results:
    for j in range(len(powers)):
        b[j] = b[j]+powers[j]
        
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(5,6))
for label, img, ax in zip(image_names, b, axes[0][:]):
    ax.imshow(img)
    ax.axis('off')

'''
for label, filt, ax in zip(image_names, b, axes[1][:]):
    ax.imshow(filt)
    ax.axis('off')
'''

plt.show()
