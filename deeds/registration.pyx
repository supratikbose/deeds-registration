from libcpp cimport bool

import SimpleITK as sitk
import numpy as np

cimport numpy as np

cdef extern from "libs/deedsBCV0.h":
    void deeds(float *im1, float *im1b, float *warped1_out,
        float *flow_flattened_out, 
        float *flow_W_out, float *flow_V_out, float *flow_U_out,   
        int *depth_out, int *height_out, int *width_out, 
        int depth_in, int height_in, int width_in, 
        bool defVectorResampledToVolume_in,
        float alpha, int levels, bool verbose)
    # void _deeds(int *depth_out)

def deeds_cpp(np.ndarray[np.float32_t, ndim=1] moving,
              np.ndarray[np.float32_t, ndim=1] fixed,
              np.ndarray[np.float32_t, ndim=1] moved,
              np.ndarray[np.float32_t, ndim=1] flow_flattened_out,
              np.ndarray[np.float32_t, ndim=1] flow_W_out,  
              np.ndarray[np.float32_t, ndim=1] flow_V_out,  
              np.ndarray[np.float32_t, ndim=1] flow_U_out,        
              shape, 
              defVectorResampledToVolume_in, 
              alpha, 
              level, 
              verbose):

    cdef int depth_out =  0 
    cdef int height_out = 0 
    cdef int width_out =  0 
    cdef int *p_depth_out = &depth_out
    cdef int *p_height_out = &height_out
    cdef int *p_width_out = &width_out

    deeds(&moving[0], &fixed[0], &moved[0], 
        &flow_flattened_out[0],
        &flow_W_out[0],  &flow_V_out[0], &flow_U_out[0],
        p_depth_out,  p_height_out, p_width_out, 
        shape[0], shape[1], shape[2], 
        defVectorResampledToVolume_in,
        alpha, level, verbose)
    return depth_out, height_out, width_out

def registration(moving_vol_np, fixed_vol_np, defVectorResampledToVolume_in=False, alpha=1.6, levels=5, verbose=True):
    moving_np = moving_vol_np.copy() #to_numpy(moving)
    fixed_np = fixed_vol_np.copy() #to_numpy(fixed)

    origin_type = moving_np.dtype
    shape = moving_np.shape

    fixed_np = fixed_np.flatten().astype(np.float32)
    moving_np = moving_np.flatten().astype(np.float32)
    moved_np = np.zeros(moving_np.shape).flatten().astype(np.float32)  

    inputSize = shape[0] * shape[1] * shape[2]
    flow_flattened_out_np    = np.zeros(3*inputSize, np.float32)
    flow_W_out_flatten  = np.zeros(inputSize, np.float32)
    flow_V_out_flatten  = np.zeros(inputSize, np.float32)
    flow_U_out_flatten  = np.zeros(inputSize, np.float32)
    
    depth_out, height_out, width_out = deeds_cpp( 
        moving_np,
        fixed_np, 
        moved_np,
        flow_flattened_out_np, 
        flow_W_out_flatten, 
        flow_V_out_flatten, 
        flow_U_out_flatten, 
        shape, 
        defVectorResampledToVolume_in,
        alpha, levels, verbose)

    #Check defVec dimension
    if True==defVectorResampledToVolume_in:
        assert depth_out==shape[0] and height_out==shape[1] and width_out==shape[2], \
            f' Unexpected output dimension.'
        defVecSize = inputSize
        defVecShape = shape
    else:
        defVecSize = depth_out *  height_out * width_out #depth_out.item() *  height_out.item() * width_out.item()
        defVecShape = tuple([depth_out, height_out, width_out]) #tuple(depth_out.item(), height_out.item(), width_out.item())
        flow_flattened_out_np = flow_flattened_out_np[:3*defVecSize]
        flow_W_out_flatten = flow_W_out_flatten[:defVecSize]
        flow_V_out_flatten = flow_V_out_flatten[:defVecSize]
        flow_U_out_flatten = flow_U_out_flatten[:defVecSize]

    #Reshape to 3D
    moved_np = np.reshape(moved_np, shape).astype(origin_type)
    flow_W_np = np.reshape(flow_W_out_flatten, defVecShape)
    flow_V_np = np.reshape(flow_V_out_flatten, defVecShape)
    flow_U_np = np.reshape(flow_U_out_flatten, defVecShape)
    
    
    # flow_3channel_np = np.reshape(flow_3channel_np, flow_3channel_shape)#.astype(origin_type)
    flow_3channel_np = np.stack([flow_U_np, flow_V_np, flow_W_np], axis=0)

    # moved_vol_np = moved_np.copy() #to_sitk(moved_np, ref_img=fixed)
    return moved_np.copy(), flow_3channel_np.copy(), flow_flattened_out_np.copy(), defVecShape


def to_numpy(img):
    result = sitk.GetArrayFromImage(img)

    return result


def to_sitk(img, ref_img=None):
    img = sitk.GetImageFromArray(img)

    if ref_img:
        img.CopyInformation(ref_img)

    return img



