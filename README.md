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
from deeds import registration
import SimpleITK as sitk

fixed_vol_np = sitk.GetArrayFromImage(sitk.ReadImage(fixed_PATH))
moving_vol_np = sitk.GetArrayFromImage(sitk.ReadImage(moving_PATH))

#In the return value  moved_vol_np  is single channel np array  and flow_3channel_np is 3 channel numpy array

moved_vol_np, flow_3channel_np = registration(fixed_vol_np, moving_vol_np)
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
