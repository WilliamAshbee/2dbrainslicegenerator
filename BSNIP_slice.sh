#!/bin/bash
​
module load Image_Analysis/FSL6.0.2
​
rm -r /data/users2/yxiao11/tem
rm -r /data/users2/yxiao11/deface_dataset
rm -r /data/users2/yxiao11/mask
rm -r /data/users2/yxiao11/mri_data
​
rm -r /data/users2/yxiao11/data/BSNIP/deface_dataset
rm -r /data/users2/yxiao11/data/BSNIP/mask
rm -r /data/users2/yxiao11/data/BSNIP/mri_data
#--------------------------------------------------
mkdir /data/users2/yxiao11/tem
mkdir /data/users2/yxiao11/deface_dataset
mkdir /data/users2/yxiao11/mask
mkdir /data/users2/yxiao11/mri_data
​
mkdir /data/users2/yxiao11/data/BSNIP/deface_dataset
mkdir /data/users2/yxiao11/data/BSNIP/mask
mkdir /data/users2/yxiao11/data/BSNIP/mri_data
​
​
foo () {
    start=`date +%s.%N`
    name=`echo $1 |rev| cut -c13-21 | rev`
    echo $name
    cp $1 /data/users2/yxiao11/mri_data/$name.nii
    fslreorient2std $1 /data/users2/yxiao11/tem/$name.nii.gz
    fsl_deface /data/users2/yxiao11/tem/$name.nii.gz /data/users2/yxiao11/deface_dataset/$name -d /data/users2/yxiao11/mask/$name
    
    python /data/users2/yxiao11/data/code/get_slices_to_npy.py -method mask -in_path /data/users2/yxiao11/deface_dataset/$name.nii.gz -ot_path /data/users2/yxiao11/data/BSNIP/deface_dataset/$name.npy
    python /data/users2/yxiao11/data/code/get_slices_to_npy.py -method mask -in_path /data/users2/yxiao11/mask/$name.nii.gz -ot_path /data/users2/yxiao11/data/BSNIP/mask/$name.npy
    python /data/users2/yxiao11/data/code/get_slices_to_npy.py -method mri -in_path /data/users2/yxiao11/mri_data/$name.nii -ot_path /data/users2/yxiao11/data/BSNIP/mri_data/$name.npy
    
    rm /data/users2/yxiao11/mri_data/$name.nii
    rm /data/users2/yxiao11/mask/$name.nii.gz
    rm /data/users2/yxiao11/deface_dataset/$name.nii.gz
    rm /data/users2/yxiao11/tem/$name.nii.gz
    end=`date +%s.%N`
    echo "$end - $start" | bc -l
}
​
export -f foo
​
ls /data/qneuromark/Data/BSNIP/RAW/*/*/anat/T1.nii | parallel foo
