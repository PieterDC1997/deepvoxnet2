#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
from deepvoxnet2.keras.metrics import dice_coefficient

def visualize_output(array1, array2, array3=None, patient_id='Undefined', vmin=None, vmax=None):
    array2 = np.where(array2 < 0.5, 0, 1)
    if array3 is not None:
        array3 = np.where(array3 < 0.5, 0, 1)
        Dice = np.sum(array2[array3==1])*2.0 / (np.sum(array2) + np.sum(array3))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 8))
        
        Dice = np.sum(array2[array3==1])*2.0 / (np.sum(array2) + np.sum(array3))
        fig.suptitle('Model segmentation output\n{0}\nDice = {1}'.format(patient_id, round(Dice,2)), fontsize = 16)
        
        index = 0
        img1 = ax1.imshow(array1[:, :, index].T, cmap = "gray", vmin=vmin, vmax=vmax)
        img2 = ax2.imshow(array2[:, :, index].T, cmap = "gray", vmin=np.min(array2), vmax=np.max(array2)) 
        img3 = ax3.imshow(array3[:, :, index].T, cmap = "gray", vmin=np.min(array3), vmax=np.max(array3)) 
        
        ax1.set_title('Image')
        ax2.set_title('Model output')
        ax3.set_title('Ground truth')
        
        
        def update_imgs3(event):
            nonlocal index
            if event.key == 'up' or event.button == 'up':
                index = (index + 1) % array1.shape[2]
            elif event.key == 'down' or event.button == 'down':
                index = (index - 1) % array1.shape[2]
            img1.set_array(array1[:, :, index].T)
            img2.set_array(array2[:, :, index].T)
            img3.set_array(array3[:, :, index].T)
            fig.canvas.draw()

        fig.canvas.mpl_connect('scroll_event', update_imgs3)
        fig.canvas.mpl_connect('key_press_event', update_imgs3)

        plt.show();
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        
        fig.suptitle('Model segmentation output\n{0}'.format(patient_id), fontsize = 16)
    
        index = 0
        img1 = ax1.imshow(array1[:, :, index].T, cmap = "gray", vmin=vmin, vmax=vmax) 
        img2 = ax2.imshow(array2[:, :, index].T, cmap = "gray", vmin=np.min(array2), vmax=np.max(array2))
        
        ax1.set_title('Image')
        ax2.set_title('Model output')
        
        def update_imgs2(event):
            nonlocal index
            if event.key == 'up' or event.button == 'up':
                index = (index + 1) % array1.shape[2]
            elif event.key == 'down' or event.button == 'down':
                index = (index - 1) % array1.shape[2]
            img1.set_array(array1[:, :, index].T)
            img2.set_array(array2[:, :, index].T)
            fig.canvas.draw()
    
        fig.canvas.mpl_connect('scroll_event', update_imgs2)
        fig.canvas.mpl_connect('key_press_event', update_imgs2)
    
        plt.show();

def visualize_output_overlay(array1, array2, array3=None, patient_id='Undefined', vmin=None, vmax=None):
    array2 = np.where(array2 < 0.5, 0, 1)
    if array3 is not None:
        array3 = np.where(array3 < 0.5, 0, 1)
        Dice = np.sum(array2[array3==1])*2.0 / (np.sum(array2) + np.sum(array3))
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.suptitle('Model segmentation output\n{0}\nDice = {1}'.format(patient_id, round(Dice,2)), fontsize = 16)


        index = 0
        combined_array = array1[:, :, index].T

        cmap_array = 'gray'  # Colormap for the first array
        cmap_overlay1 = 'Greens'  # Colormap for the overlay
        cmap_overlay2 = 'Reds'

        img1 = ax.imshow(combined_array, cmap=cmap_array, vmin = vmin, vmax = vmax)

        overlay_alpha = 0.2  # Overlay transparency (0.0 - fully transparent, 1.0 - fully opaque)
        img2 = ax.imshow(array2[:, :, index].T, cmap=cmap_overlay1, alpha=overlay_alpha, vmin=np.min(array2), vmax=np.max(array2)) 
        img3 = ax.imshow(array3[:, :, index].T, cmap=cmap_overlay2, alpha=overlay_alpha, vmin=np.min(array3), vmax=np.max(array3)) 
        
        def update_img3(event):
            nonlocal index
            if event.key == 'up' or event.button == 'up':
                index = (index + 1) % array1.shape[2]
            elif event.key == 'down' or event.button == 'down':
                index = (index - 1) % array1.shape[2]
            combined_array = array1[:, :, index].T
            img1.set_array(combined_array)
            img2.set_array(array2[:, :, index].T)
            img3.set_array(array3[:, :, index].T)
            fig.canvas.draw()

        fig.canvas.mpl_connect('scroll_event', update_img3)
        fig.canvas.mpl_connect('key_press_event', update_img3)

        # Add colorbars
        cbar2 = fig.colorbar(img2, ax=ax)
        cbar2.set_label('Model output', fontsize = 14)
        cbar3 = fig.colorbar(img3, ax=ax)
        cbar3.set_label('Ground Truth', fontsize = 14)
        #cbar1 = ax.collections[0].colorbar
        #cbar1.remove()
        plt.show();
    else:      
        
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.suptitle('Model segmentation output\n{0}'.format(patient_id), fontsize = 16)
    
        index = 0
        combined_array = array1[:, :, index].T
    
        cmap_array = 'gray'  # Colormap for the first array
        cmap_overlay = 'Greens'  # Colormap for the overlay
    
        img1 = ax.imshow(combined_array, cmap=cmap_array, vmin = vmin, vmax = vmax)
    
        overlay_alpha = 0.2  # Overlay transparency (0.0 - fully transparent, 1.0 - fully opaque)
        img2 = ax.imshow(array2[:, :, index].T, cmap=cmap_overlay, alpha=overlay_alpha, vmin=np.min(array2), vmax=np.max(array2)) 
    
        def update_img2(event):
            nonlocal index
            if event.key == 'up' or event.button == 'up':
                index = (index + 1) % array1.shape[2]
            elif event.key == 'down' or event.button == 'down':
                index = (index - 1) % array1.shape[2]
            combined_array = array1[:, :, index].T
            img1.set_array(combined_array)
            img2.set_array(array2[:, :, index].T)
            fig.canvas.draw()
    
        fig.canvas.mpl_connect('scroll_event', update_img2)
        fig.canvas.mpl_connect('key_press_event', update_img2)
    
        # Add colorbars
        cbar2 = fig.colorbar(img2, ax=ax)
        cbar2.set_label('Model output', fontsize = 14)
        plt.show();
   