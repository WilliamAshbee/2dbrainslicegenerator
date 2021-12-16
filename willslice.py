import nibabel as nib
#dir = '/data/hcp-plis/hdd05/101410/T1w/101410/mri'

import matplotlib.pyplot as plt

#img = nib.load('/data/hcp-plis/hdd05/101410/T1w/101410/mri/T1w_hires.norm.nii.gz')
#img = img.get_fdata()

# plt.imshow(img2)
# plt.savefig('img2.png')
# plt.clf()
# plt.imshow(img3)
# plt.savefig('img3.png')
# plt.clf()

import os
# assign directory
directory = '/data/hcp-plis/hdd05/101410/T1w/101410/mri/'
 
# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
        
    try:
        img = nib.load(f)
        img = img.get_fdata()
        img1 = img[img.shape[0]//2,:,:]
        img2 = img[:,img.shape[1]//2,:]
        img3 = img[:,:,img.shape[2]//2]


        plt.imshow(img1)
        plt.savefig(filename+'.1.png')
        plt.clf()
        plt.imshow(img2)
        plt.savefig(filename+'.2.png')
        plt.clf()
        plt.imshow(img3)
        plt.savefig(filename+'.3.png')
        plt.clf()

        # checking if it is a file
        if os.path.isfile(f):
            print(f)
    except:
        print('file ', f, 'through an exception.')
    