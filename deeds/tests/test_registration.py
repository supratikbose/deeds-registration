import unittest
import nibabel as nib

from ..registration import registration


class TestStringMethods(unittest.TestCase):

    def test_deeds_registration(self):
        fixed = nib.load('/home/wd974888/Documents/defVector_FullScan_MnBInput_onlyCurrent_sharedFlow_noWtMSE_2247775/fixedScan_16.nii.gz').get_fdata() #load_nifty('samples/fixed.nii.gz')
        moving = nib.load('/home/wd974888/Documents/defVector_FullScan_MnBInput_onlyCurrent_sharedFlow_noWtMSE_2247775/movingScan_14.nii.gz').get_fdata() #load_nifty('samples/moving.nii.gz')
        moved, flow = registration(fixed, moving)
