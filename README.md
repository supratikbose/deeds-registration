# DEnse Displacement Sampling - deformable image registration

This package provides Python wrapper around [DEEDS](https://github.com/mattiaspaul/deedsBCV), an efficient version for 3D discrete deformable image registration which is reaching the highest accuracy in several benchmarks [[1]](https://pubmed.ncbi.nlm.nih.gov/27254856/)[[2]](https://arxiv.org/abs/2109.11572) and serves as a good baseline for new solutions. Main changes from the parent repository (a) inputs and outputs are now numpy array instead of sitk images and (b) flow (deformation) vector is also given out as 3 channel np volume. TODO - Unit Tesh to be updated.

## Referencing and citing
If you use this implementation or parts of it please cite:
 
>"MRF-Based Deformable Registration and Ventilation Estimation of Lung CT."
 by Mattias P. Heinrich, M. Jenkinson, M. Brady and J.A. Schnabel
 IEEE Transactions on Medical Imaging 2013, Volume 32, Issue 7, July 2013, Pages 1239-1248
 http://dx.doi.org/10.1109/TMI.2013.2246577
 
 and
 
>"Multi-modal Multi-Atlas Segmentation using Discrete Optimisation and Self-Similarities"
 by Mattias P. Heinrich, Oskar Maier and Heinz Handels
 VISCERAL Challenge@ ISBI, Pages 27-30 2015
 http://ceur-ws.org/Vol-1390/visceralISBI15-4.pdf
 
## Installation
```
pip install git+https://github.com/supratikbose/deeds-registration
```

## Usage
```
#See usage.ipynb

#Install : pip install git+https://github.com/supratikbose/deeds-registration
from deeds import registration
import SimpleITK as sitk
#Volumes are expected to be have identical dimension and identical isometric pixel spacing 
fixed_vol_np = sitk.GetArrayFromImage(sitk.ReadImage(fixed_PATH))
moving_vol_np = sitk.GetArrayFromImage(sitk.ReadImage(moving_PATH))
#Invoke Deeds
moved_vol_np, flow_3channelUVW_np, flow_flattened_out_np, defVecShape =\
    registration(moving_vol_np, fixed_vol_np, defVectorResampledToVolume_in=True,alpha=1.6,\
    levels=5, verbose=True)

#Interpreting input and result
#In the input and result volumes, i.e., in  moving_vol_np, fixed_vol_np and in 
#moved_vol_np we assume the 1st dimension is depth(Z of size o) followed by 
#height(Y of size n) and finally width(X of size m). Deformation vector is 
# also given out in 3-channel flow_3channelUVW_np where channels are of the 
# order [U,V,W]

#Examining deeds/libs/dataCostD.h/warpAffine() we interprete U,V,W as follows:
# During warping, fastest (innermost) loop is along W (size m); next 
# along H (size n) and finally along D (size o). Moreover
#U modifies x with x limit betwee 0 and N!! so U is displacement along 2nd dimension!! 
#i.e., Height
# V modifies y with y limit betwee 0 and M!! so V is displacement along 3rd dimension!! 
#i.e., Width
# W modifies z with z limit betwee 0 and O!! so W is displacement along 1st dimension!! 
#i.e., Depth

#If defVectorResampledToVolume_in is false, it is of lower resolution than the input / output 
# volume. It is the same output whose flattened version is deformations.dat file.
# If defVectorResampledToVolume_in flag is true, flow_3channelUVW_np, and 
# flow_flattened_out_np are in full volume dimension.


#Helper function to deform volume using DVF returned by DEEDS
#Given deformation vector output flow_3channelUVW_np at full resolution (i.e. 
# defVectorResampledToVolume_in=True), one can use MONAI (1.1) to warp the 
#moving image to generate moved image outside Deeds as below:

def deformUsingDeedsDefVecAndMonaiWarp(vol_M_DHW, flow_3channelUVW_np):

    import torch
    from monai.networks.blocks import Warp as monai_warp
    a_monai_warp_transformer = monai_warp(mode='bilinear', padding_mode='border')
    
    vol_M_torch = torch.from_numpy(vol_M_DHW.astype('float32'))
    #Get u,v,w components from deeds_defVec
    u= torch.from_numpy(flow_3channelUVW_np[0,...].astype('float32'))
    v= torch.from_numpy(flow_3channelUVW_np[1,...].astype('float32'))
    w= torch.from_numpy(flow_3channelUVW_np[2,...].astype('float32'))

    #Add batch channel
    vol_M_torch_bc = vol_M_torch.unsqueeze(0).unsqueeze(0)
    #NOTE Since volume is in DHW, and from deeds we know w along D, u along H and v along w 
    #we need to pack deformation field in same DHW order. We also add batch and channel
    ddf_bc = torch.stack([w,u,v], dim=0).unsqueeze(0)

    #Apply Monai warp
    vol_MStarLocalDeeds_monai_torch = a_monai_warp_transformer(vol_M_torch_bc,  ddf_bc)
    return vol_MStarLocalDeeds_monai_torch.squeeze().detach().cpu().numpy().astype('float')

#Use  deformation vector returned by DEEDS to deform moving volume "locally" 
# and check mean square error (mse)

#Install : pip install sewar
from sewar.full_ref import mse
locally_moved_vol_np = deformUsingDeedsDefVecAndMonaiWarp(vol_M_DHW=moving_vol_np,\
 flow_3channelUVW_np=flow_3channelUVW_np)
#MSE before deformation
mse_b4Def = mse(fixed_vol_np,  moving_vol_np)
print(f'mse before deformation {round(mse_b4Def,6)}') 
#mse_b4Def 0.002815
#MSE after deformation using DEEDS
mse_afterDef_Deeds = mse(fixed_vol_np,  moved_vol_np)
print(f'mse after deformation using Deeds {round(mse_afterDef_Deeds,6)}') 
#mse_afterDef_Deeds 0.000678
#MSE after local deformation using DVF returned by DEEDS
mse_afterLocalDef_Deeds_monai = mse(fixed_vol_np,  locally_moved_vol_np)
print(f'mse_afterLocalDef_Deeds_monai {round(mse_afterLocalDef_Deeds_monai,6)}') 
#mse_afterLocalDef_Deeds_monai 0.000678
#MSE between two deformed volumes (by DEEDS and by local deformation using DEEDS DVF)
mse_between_two_methods = mse(moved_vol_np,  locally_moved_vol_np)
print(f' between two deformed volumes {round(mse_between_two_methods,6)}') 
#mse_between_two_methods 0.0

```

## Prerequesities
Input image volumes must be numpy array having the same dimensions

## Development
Build:
```
python setup.py build_ext --inplace
```
## Test
Test ( The -m flag is necessary to  import from "local" registration module.):
```
python -m unittest 
```
