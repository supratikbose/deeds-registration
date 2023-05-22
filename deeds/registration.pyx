from libcpp cimport bool

import SimpleITK as sitk
import numpy as np

cimport numpy as np

cdef extern from "libs/deedsBCV0.h":
    void deeds(float *im1, float *im1b, float *warped1, float *flow_U, float *flow_V, float *flow_W, int m, int n, int o, float alpha, int levels, bool verbose)


def deeds_cpp(np.ndarray[np.float32_t, ndim=1] fixed,
              np.ndarray[np.float32_t, ndim=1] moving,
              np.ndarray[np.float32_t, ndim=1] moved,
              np.ndarray[np.float32_t, ndim=1] flow_U,
              np.ndarray[np.float32_t, ndim=1] flow_V,
              np.ndarray[np.float32_t, ndim=1] flow_W,
              shape, alpha, level, verbose):
    return deeds(&moving[0], &fixed[0], &moved[0], &flow_U[0], &flow_V[0], &flow_W[0],
                  shape[2], shape[1], shape[0],
                  alpha, level, verbose)


def registration(fixed_vol_np, moving_vol_np, alpha=1.6, levels=5, verbose=True):
    fixed_np = fixed_vol_np.copy() #to_numpy(fixed)
    moving_np = moving_vol_np.copy() #to_numpy(moving)

    origin_type = moving_np.dtype

    shape = moving_np.shape
    
    # flow_1channel_np=np.zeros(moving_np.shape)
    # flow_3channel_np=np.repeat(flow_1channel_np[np.newaxis,:,:,:], 3, axis=0)
    # flow_3channel_shape=flow_3channel_np.shape 
    #print(f'flow_3channel_shape b4 flatten: {flow_3channel_shape}')   
    flow_U_np=np.zeros(shape)
    flow_V_np=np.zeros(shape)
    flow_W_np=np.zeros(shape)
    
    fixed_np = fixed_np.flatten().astype(np.float32)
    moving_np = moving_np.flatten().astype(np.float32)
    moved_np = np.zeros(moving_np.shape).flatten().astype(np.float32)
    #flow_3channel_np=flow_3channel_np.flatten().astype(np.float32)
    flow_U_np=flow_U_np.flatten().astype(np.float32)
    flow_V_np=flow_V_np.flatten().astype(np.float32)
    flow_W_np=flow_W_np.flatten().astype(np.float32)

    deeds_cpp(fixed_np, moving_np, moved_np, flow_U_np, flow_V_np, flow_W_np, shape, alpha, levels, verbose)

    moved_np = np.reshape(moved_np, shape).astype(origin_type)
    flow_U_np = np.reshape(flow_U_np, shape)
    flow_V_np = np.reshape(flow_V_np, shape)
    flow_W_np = np.reshape(flow_W_np, shape)
    # flow_3channel_np = np.reshape(flow_3channel_np, flow_3channel_shape)#.astype(origin_type)
    flow_3channel_np = np.stack([flow_U_np, flow_V_np, flow_W_np], axis=0)

    # moved_vol_np = moved_np.copy() #to_sitk(moved_np, ref_img=fixed)
    return moved_np.copy(), flow_3channel_np.copy()


def to_numpy(img):
    result = sitk.GetArrayFromImage(img)

    return result


def to_sitk(img, ref_img=None):
    img = sitk.GetImageFromArray(img)

    if ref_img:
        img.CopyInformation(ref_img)

    return img
