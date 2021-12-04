import nibabel as nib
from skimage import measure
import numpy as np
import argparse
import os
import cv2
​
​
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-in_path', type=str,
                    help='the input path of nii file')
parser.add_argument('-ot_path', type=str,
                    help='the output path of nii file')
parser.add_argument('-method', type=str, default=None,
                    help='make sure to select the middle slice spatial direction')
​
​
args = parser.parse_args()
​
image = nib.load(args.in_path)
image = image.get_fdata()
​
# --------this is for ABCD file-----------
# slices = image[88,:,:]
# ---------------------------------------
​
# ---------this is for BSNIP file-----------
​
if args.method == "mask":
    slices = image[80, :, :]
    slices = cv2.resize(slices, dsize=(256, 256))
elif args.method == "mri":
    slices = image[:, :, 80]
    slices = cv2.resize(slices, dsize=(256, 256))
    slices = np.flipud(slices)

# ------------------------------------
​
# ---------this is for FBIRN file-----------
​
# if args.method == "mask":
#     slices = image[88,:,:]
#     # slices = cv2.resize(slices, dsize=(256,256))
# elif args.method == "mri":
#     slices = image[:,:,88]
#     # slices = cv2.resize(slices, dsize=(256,256))
#     slices = np.rot90(slices, k=2)

# ------------------------------------
​
​
np.save(args.ot_path, slices)
​
# file_name = os.listdir(args.in_path)[0]
​
# image = nib.load(args.in_path + file_name)
# image = image.get_fdata()
​
# slices = image[88,:,:]
​
# np.save(args.ot_path + file_name[0:15], slices)
