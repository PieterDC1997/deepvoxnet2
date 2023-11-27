#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:55:10 2023

@author: pieter
"""

import os
import nibabel as nib
import numpy as np
    
    
def orient_to_lps_3d(file_path):
    # Load the NIfTI file
    nii_img = nib.load(file_path)
    
    data = nii_img.get_fdata()
    affine = nii_img.affine

    # Check the current image orientation
    current_orientation = nib.aff2axcodes(affine)

    # Define the target LPS orientation
    target_orientation = ('L', 'P', 'S')

    # Flip the data if necessary
    if current_orientation[0] != target_orientation[0]:
        data = np.flip(data, axis=0)
        affine[0] *= -1.0  # Update the affine matrix for flipped axis

    if current_orientation[1] != target_orientation[1]:
        data = np.flip(data, axis=1)
        affine[1] *= -1.0

    if current_orientation[2] != target_orientation[2]:
        data = np.flip(data, axis=2)
        affine[2] *= -1.0
    
    data = data.squeeze()

    # Create a new NIfTI image with the LPS-oriented data and updated affine matrix
    lps_nii_img = nib.Nifti1Image(data, affine)

    # Update the header with the correct orientation codes
    lps_nii_img.header.set_qform(affine, code='scanner')
    lps_nii_img.header.set_sform(affine, code='scanner')

    # Get the original filename without extension
    return lps_nii_img
    

    