import unittest
import nibabel as nib
import pathlib
import numpy as np
from sewar.full_ref import mse

from ..registration import registration


class TestStringMethods(unittest.TestCase):
    def test_deeds_registration(self):
        fixed = nib.load('samples/fixed.nii.gz').get_fdata() 
        moving = nib.load('samples/moving.nii.gz').get_fdata() 
        vol_MStar_Deeds, deformVec_Deeds = registration(fixed, moving)
        mse_b4Def = mse(fixed, moving)
        mse_afterDef = mse(fixed, vol_MStar_Deeds)
        print(f"MSE B4:{round(mse_b4Def,4)} after: {round(mse_afterDef,4)}")
        np.savez_compressed('samples/deedsResultTmp.npz',vol_MStar_Deeds=vol_MStar_Deeds, deformVec_Deeds=deformVec_Deeds) 

    # def test_deeds_registration(self):
    #     usePreCreatedSynthetic=True

    #     if False==usePreCreatedSynthetic:
    #         print(f'Creating synthetic volumes')
    #         shift_X_d1 = 20 
    #         shift_Y_d0 = 0
    #         shift_Z_d2 = 0
    #         startLoc_M=[10,10,10]
    #         largeCubeDim=[256, 256, 256]
    #         inner_cubeDim=[140,160,180]
    #         startVoxVal_float=1000.5
    #         endVoxVal_float=2000.5
    #         vol_M, vol_F = createSyntheticVolumes(shift_X_d1, shift_Y_d0, shift_Z_d2, startLoc_M, largeCubeDim, inner_cubeDim, startVoxVal_float,  endVoxVal_float)
    #         print(f'Saving synthetic volumes')
    #         refImageFilePath = 'samples/fixed.nii.gz'
    #         refImage_affine = nib.load(str(refImageFilePath)).affine
    #         nib.save(nib.Nifti1Image(vol_M, refImage_affine), filename= str('samples/synthetic_vol_M.nii.gz'))
    #         nib.save(nib.Nifti1Image(vol_F, refImage_affine), filename= str('samples/synthetic_vol_F.nii.gz'))
    #     else:
    #         print(f'Reading pre-created synthetic volumes')
    #         vol_M = nib.load('samples/synthetic_vol_M.nii.gz').get_fdata().astype('float32')
    #         vol_F = nib.load('samples/synthetic_vol_F.nii.gz').get_fdata().astype('float32')
    #     print(f'Invoking deeds')
    #     vol_MStar_Deeds, deformVec_Deeds = registration(vol_F, vol_M)
    #     mse_b4Def = mse(vol_F, vol_M)
    #     mse_afterDef = mse(vol_F, vol_MStar_Deeds)
    #     print(f"MSE B4:{round(mse_b4Def,4)} after: {round(mse_afterDef,4)}")
    #     np.savez_compressed(str('samples/syntheticVol_deedsResultTmp.npz'),vol_MStar_Deeds=vol_MStar_Deeds, deformVec_Deeds=deformVec_Deeds)

def createSyntheticVolumes(shift_X_d1 = 20, shift_Y_d0 = 0, shift_Z_d2 = 0,
        startLoc_M=[10,10,10], largeCubeDim=[256, 256, 256], inner_cubeDim=[140,160,180], startVoxVal_float=1000.5,  endVoxVal_float=2000.5):
    print(f'Creating synthetic moving Volume')
    print(f'startLoc_M {startLoc_M}')
    vol_M = np.zeros(tuple(largeCubeDim),'float32')
    vol_M = placeCubeInVolume(vol_M, cubeDim_list=inner_cubeDim, startLoc_list=startLoc_M, startVoxVal_float=startVoxVal_float, endVoxVal_float=endVoxVal_float, verbose=True)
    print(f'Creating synthetic fixed Volume by shifting previous volume by shift_X_d1: {shift_X_d1} shift_Y_d0: {shift_Y_d0} shift_Z_d2: {shift_Z_d2}')
    startLoc_F=[startLoc_M[0]+shift_Y_d0, startLoc_M[1]+shift_X_d1, startLoc_M[2]+shift_Z_d2]
    print(f'startLoc_F {startLoc_F}')    
    vol_F = np.zeros(tuple(largeCubeDim),'float32')
    vol_F = placeCubeInVolume(vol_F, cubeDim_list=inner_cubeDim, startLoc_list=startLoc_F, startVoxVal_float=startVoxVal_float, endVoxVal_float=endVoxVal_float, verbose=True)
    return vol_M, vol_F

def placeCubeInVolume(volume_np, cubeDim_list, startLoc_list, startVoxVal_float, endVoxVal_float, verbose=False):
    """
    cubeDim_list:(height, width, depth)
    """
    if verbose:
        print(f'volume_np.shape: {volume_np.shape} volume_np type: {type(volume_np)}')
        print(f'cubeDim_list: {cubeDim_list} type: {type(cubeDim_list)}')
        print(f'startLoc_list: {startLoc_list} type: {type(startLoc_list)}')
        print(f'startVoxVal_float: {startVoxVal_float} type: {type(startVoxVal_float)}')
        print(f'endVoxVal_float: {endVoxVal_float} type: {type(endVoxVal_float)}')

    endLoc_list=(np.array(startLoc_list,'int') + np.array(cubeDim_list, 'int') + np.array([-1,-1,-1], 'int')).tolist()
    if verbose:
        print(f'endLoc_list: {endLoc_list} type: {type(endLoc_list)}')

    assert endLoc_list[0] < volume_np.shape[0] and endLoc_list[1] < volume_np.shape[1] and endLoc_list[2] < volume_np.shape[2], \
        f'adjust  dimensions of volume_np, cubeDim_list or startLoc_list so that endLoc_list is within volume'
    
    dist_diagonal = np.linalg.norm(np.array(cubeDim_list, 'float32') - np.array([1,1,1], 'float32')).item() #Subtract [1,1,1] to get voxel difference
    deltaVoxelperUnitDistance = (endVoxVal_float - startVoxVal_float)/dist_diagonal
    if verbose:
        print(f'dist_diagonal: {dist_diagonal} deltaVoxelperUnitDistance: {deltaVoxelperUnitDistance}')

    volume_np=np.zeros_like(volume_np).astype('float32')
    volume_np[startLoc_list[0], startLoc_list[1], startLoc_list[2]]= startVoxVal_float
    for d_index in range(cubeDim_list[2]):
        for h_index in range(cubeDim_list[0]):
            for w_index in range(cubeDim_list[1]):
                location = [h_index+startLoc_list[0], w_index+startLoc_list[1], d_index+startLoc_list[2]]
                distFromStartVoxel = np.linalg.norm(np.array([h_index, w_index, d_index], 'float32')).item() 
                voxelAtLocation = startVoxVal_float + deltaVoxelperUnitDistance * distFromStartVoxel
                volume_np[location[0], location[1], location[2]]= voxelAtLocation
    return volume_np