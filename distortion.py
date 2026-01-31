import numpy as np
import numpy.typing as npt
from typing import Tuple
import nibabel as nib
import os
from scipy.interpolate import RBFInterpolator as RBF

from preprocessing import separate_SMS, gaussian_filter, smooth_mask_edges
from utils import mask_array, displayImg
from skimage.morphology import area_opening   


def _DPhTE12(Mags: npt.NDArray[np.float_], Phs: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Computes the phase accumulated from TE1 to TE2 due to main-field inhomogeneities.
    - The complex images are averaged at each TE for improving the SNR.
    - The phase evolution from TE1 to TE2 is obtained by taking the argument of
    the complex division C_Img_avgTE2 / C_Img_avgTE1

    Parameters
    ----------
    Mags : npt.NDArray[np.float_]
        Time series of 3D magnitude images
    Phs : npt.NDArray[np.float_]
        Time series of 3D phase images. Must be scaled to [-pi,pi]

    Returns
    -------
    npt.NDArray[np.float_]
        Average phase accumulated between TE1 and TE2
    """

    assert np.amin(Phs) >= - np.pi and np.amax(Phs) <= np.pi 
        
    C_Imgs_TE1 = Mags[..., 0::2] * np.exp( 1j*Phs[..., 0::2] )
    C_Imgs_TE2 = Mags[..., 1::2] * np.exp( 1j*Phs[..., 1::2] )
    
    print(f"[_DPhTE12 in distortion.py] Averaging {C_Imgs_TE1.shape[3]} complex images to compute the EPI-based field map\n", flush=1)
    
    avg_C_Imgs_TE1_m = np.mean(C_Imgs_TE1, axis=3)
    avg_C_Imgs_TE2_m = np.mean(C_Imgs_TE2, axis=3)
    
    deltaPh = np.angle( avg_C_Imgs_TE2_m / avg_C_Imgs_TE1_m )

    return deltaPh


def _spatialUW(Mags_path, Ph2sUW_path, mask_path="", PhsUW_path=None):
    """
    Runs PRELUDE to perform spatial UnWrapping of the phase image provided
    """

    if PhsUW_path is None: PhsUW_path = Ph2sUW_path.split(".nii")[0] + "_sUW.nii.gz"

    cmd = f'prelude -a {Mags_path} -p {Ph2sUW_path} -u {PhsUW_path} -s'
    if len(mask_path) > 0: cmd = f'{cmd} -m {mask_path}'
    
    os.system(cmd)

    Ph_sUW = np.array(nib.load(PhsUW_path).get_fdata())
    
    if Ph_sUW.ndim == 2: 
        Ph_sUW = Ph_sUW[:,:,None]

    return Ph_sUW


def _minimum_sum_of_phase(Ph: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Adds/subtracts multiples of 2pi to get the phase image with the minimum sum of abs(phase)

    Parameters
    ----------
    Ph : npt.NDArray[np.float_]
        3D Phase image (in rad)

    Returns
    -------
    npt.NDArray[np.float_]
        Phase image with the minimum sum of abs(phase) in each slice
    """

    Ph_minSum = np.ma.zeros_like(Ph)

    for slc in range(Ph.shape[2]):

        Ph_slc = Ph[:,:,slc].compressed()
        
        prevSum = np.sum(abs(Ph_slc))

        # 1. Test *adding* multiples of 2pi
        k = +1
        newSum = np.sum(abs(Ph_slc + k*2*np.pi))
        while newSum < prevSum:
            prevSum = newSum
            
            k += 1
            newSum = np.sum(abs(Ph_slc + k*2*np.pi))
            
        # We leave the loop when newSum > originalSum -> go back to the previous k <-> previous sum:
        k -= 1
        
        # 2. If adding multiples of 2pi made it worse -> try subtracting
        if k == 0: 
        
            k = -1
            newSum = np.sum(abs(Ph_slc + k*2*np.pi))
            while newSum < prevSum:
                prevSum = newSum
                
                k -= 1
                newSum = np.sum(abs(Ph_slc + k*2*np.pi))
                
            # Again, we leave the loop when newSum > originalSum -> go back to the previous k <-> previous sum:
            k += 1
        
        Ph_minSum[:,:,slc] = Ph[:,:,slc] + k*2*np.pi
    
    return Ph_minSum



def get_DPhTE12_sUW(Mags: npt.NDArray[np.float_], 
                    Phs: npt.NDArray[np.float_], 
                    affine_3D: npt.NDArray[np.float_], 
                    Mags_path: str, 
                    DPhTE12_path: str) -> npt.NDArray[np.float_]:
    """
    Computes the average phase accumulation between the two echo times. Removes potential phase wraps

    Parameters
    ----------
    Mags : npt.NDArray[np.float_]
        Time series of 3D magnitude images
    Phs : npt.NDArray[np.float_]
        Time series of 3D phase images. Must be scaled to [-pi,pi]
    affine_3D : npt.NDArray[np.float_]
        A 4x4 affine matrix for each slice. Needed for separating DPh images from SMS 
        acquisitions into single slices which can then be unwrapped
    Mags_path : str
        Full path of the magnitude images' nifti file. Needed for the spatial unwrapping
    DPhTE12_path : str
        Full path of the unwrapped phase difference image to be created

    Returns
    -------
    npt.NDArray[np.float_]
        Average phase accumulation between the two echo times (without phase wraps)
    """

    assert np.ma.isMA(Mags) and np.ma.isMA(Phs)

    DPh_TE12 = _DPhTE12(Mags, Phs)

    # Save the DPh img(s) as .nii so that PRELUDE can use it
    nib.save( nib.Nifti1Image(DPh_TE12.filled(0), nib.load(Mags_path).affine), DPhTE12_path )

    N_slcs = affine_3D.shape[2]

    if N_slcs == 1:
        mask_dir = os.path.join( os.path.split(Mags_path)[0], "1.Masks/")
        mask_FN = [FN for FN in os.listdir(mask_dir) if os.path.split(Mags_path)[1][:-4] in FN and "Mask" in FN]
        assert len(mask_FN) == 1, print(mask_FN)
        
        DPhTE12_sUW = _spatialUW(Mags_path, DPhTE12_path, mask_dir+mask_FN[0])
    else:
        separate_SMS(DPhTE12_path, affine_3D)

        mask_dir = Mags_path.split(".nii")[0] + "_Slcs/"

        DPhTE12_sUW_slcs_l = []

        for slc in range(N_slcs):

            Mags_slc_path    = os.path.join(Mags_path.split(".nii")[0] + "_Slcs/", os.path.split(Mags_path)[1].replace(".nii", "_Slc%d.nii" % slc))
            DPhTE12_slc_path = os.path.join(DPhTE12_path.split(".nii")[0] + "_Slcs/", os.path.split(DPhTE12_path)[1].replace(".nii", "_Slc%d.nii" % slc))

            mask_FN = [FN for FN in os.listdir(mask_dir) if os.path.split(Mags_slc_path)[1][:-4] in FN and "Mask" in FN]
            assert len(mask_FN) == 1

            DPhTE12_sUW_slc = _spatialUW(Mags_slc_path, DPhTE12_slc_path, mask_dir+mask_FN[0])

            DPhTE12_sUW_slcs_l.append(DPhTE12_sUW_slc[:,:,0])

        DPhTE12_sUW = np.stack(DPhTE12_sUW_slcs_l, axis=2)

        nib.save( nib.Nifti1Image(DPhTE12_sUW, nib.load(Mags_path).affine), DPhTE12_path.replace(".nii", "_sUW.nii.gz") )

    DPhTE12_sUW = np.ma.masked_where(Mags[:,:,:,0].mask, DPhTE12_sUW)

    DPhTE12_sUW = _minimum_sum_of_phase(DPhTE12_sUW)

    return DPhTE12_sUW



def eval_at_xEdges(im: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """Fits a thin-plate-spline to the input image and evaluates it 
    at the edges of the voxels (in the x/horziontal direction)
    """
    
    assert im.ndim == 2, print(im.shape)
           
    # Original grid
    xGrid, yGrid1 = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    
    # Upsample in x:
    xEdges, yGrid2 = np.meshgrid(np.arange(im.shape[1]+1), np.arange(im.shape[0]))
    xEdges = xEdges - 0.5
        
    # Coords of the training set:
    xGrid_train, yGrid_train = [ np.ma.masked_where(im.mask, grid).compressed() for grid in [xGrid, yGrid1] ]
    coords_train = np.stack([xGrid_train, yGrid_train], axis=-1)
       

    kName = 'thin_plate_spline'; 
    neigh = 15; # SHOULD BE 15! REMEMBER TO RESET
    s = 0 #0.1
    interpolator = RBF(coords_train, im.compressed(), kernel=kName, neighbors=neigh, smoothing=s, degree=1)

    im_interp = interpolator(np.stack([xGrid.flatten(), yGrid1.flatten()], axis=-1)).reshape(im.shape)
    im_interp = np.ma.masked_where(im.mask, im_interp)

    im_xEdges = interpolator(np.stack([xEdges.flatten(), yGrid2.flatten()], axis=-1)).reshape(xEdges.shape)
    #(Nrow, Ncol+1)
        
    return im_xEdges




def _piecewise_linear_interp(im: npt.NDArray[np.float_]):
    """Performs a 1D piece-wise linear interpolation of the 
    input image along the x (horiozontal) direction

    Parameters
    ----------
    im : npt.NDArray[np.float_]
        Image to be interpolated

    Returns
    -------
    slopes: npt.NDArray[np.float_])
    intercepts: npt.NDArray[np.float_])]
    """

    assert im.ndim == 2
    
    xEdges, yGrid = np.meshgrid(np.arange(im.shape[1]+1), np.arange(im.shape[0]))
    xEdges = xEdges - 0.5
    
    im_xEdges = eval_at_xEdges(im)
    
    slopes = im_xEdges[:,1:] - im_xEdges[:,0:-1] 
    intercepts = im_xEdges[:,0:-1] - slopes*xEdges[:,0:-1]
    
    if np.ma.is_masked(im):
        slopes, intercepts = [ np.ma.masked_where(im.mask, arr) for arr in [slopes, intercepts] ]

    return slopes, intercepts



def undistort_shift_map(dy: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Computes the pixel shift map in the UNdistorted space (dy') from the pixel shift map in the distorted space (dy)

    Parameters
    ----------
    dy : npt.NDArray[np.float_]
        Pixel shift map in the distorted image space
        
    Returns
    -------
    npt.NDArray[np.float_]
        Pixel shift map in the UNdistorted space
    """
    
    def _test_candidate(col_prime, test_col, dy_slopes, dy_intercepts):
        
        dy_prime_candidate, found = None, False
        
        a = dy_slopes[row, test_col]
        b = dy_intercepts[row, test_col]
        
        if not np.ma.is_masked(a): 
            dy_prime_candidate = (a*col_prime + b) / (1-a)
        
            if test_col-0.5 <= col_prime + dy_prime_candidate <= test_col+0.5:
                found = True
        
        return dy_prime_candidate, found
        
    assert dy.ndim == 3, print(dy.shape)
    
    dy_prime = float("nan") * np.ma.ones_like(dy)
    
    for slc in range(dy_prime.shape[2]):
    
        dy_slopes, dy_intercepts = _piecewise_linear_interp(dy[:,:,slc]) # 2D arrays
    
        dy_max = np.amax(abs(dy[:,:,slc]))
        
        for row in range(dy_prime.shape[0]):
            # Horizontal pixel shifting within each row:
            for col_prime in range(dy_prime.shape[1]): # col_prime: column in dy_prime
            
                found = False # true if we have found, in the *Distorted* space (dy), the col that corresponds to the col we are trying to fill in the UNdistorted space (dy')
                col_shift = 0 # 1st try col=col_prime (meaning that dy' < 1 voxel). 
                              # If the result is out of the y bonds of that col, we try col=col_prime +- 1, etc
                
                # --------------------------------------------------
                while not found and col_shift <= np.ceil(dy_max):
                    
                    test_col = col_prime + col_shift # in dy
                    
                    if test_col < dy_slopes.shape[1]:
                        dy_prime_candidate, found = _test_candidate(col_prime, test_col, dy_slopes, dy_intercepts)
                        
                    if not found: # If the voxel at (row,test_col) was masked in the dy or the dy_prime_candidate revealed that we were looking into the wrong col in dy
                        
                        test_col = col_prime - col_shift
                        if test_col > 0:
                            dy_prime_candidate, found = _test_candidate(col_prime, test_col, dy_slopes, dy_intercepts)
    
                    if found:
                        dy_prime[row, col_prime, slc] = dy_prime_candidate
                        
                    else:
                        col_shift += 1 # If not found, we will try the voxels 1 position further apart from col_prime
                # --------------------------------------------------                
               
                if not found: dy_prime[row, col_prime, slc] = float("nan")
                    
        dy_prime = np.ma.masked_where(np.isnan(dy_prime), dy_prime)
                  
    return dy_prime


def xGradient(im: npt.NDArray[np.float_], res_x: float) -> npt.NDArray[np.float_]:
    """Computes discrete derivatives of the input image along the x (horizontal) direction
    
    Parameters
    ----------
    im : npt.NDArray[np.float_]
        2D input image
    res_x: float
        Voxel size in the horizontal direction (in m)

    Returns
    -------
    npt.NDArray[np.float_]
        Discrete derivatives of the input image along the x (horizontal) direction (in /m)
    """
    
    assert im.ndim == 2, print(im.shape)
    
    im_xEdges = eval_at_xEdges(im)
    xgradient = np.zeros_like(im)
    
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            if not np.ma.is_masked(im[row, col]):
                
                # If this is the leftmost pixel of this row -> don't trust the fit on the left edge (it is extrapolated)
                if col == 0 or np.ma.is_masked(im[row, col-1]): 
                    xgradient[row,col] =  im[row,col+1]- im[row,col] # (im_xEdges[row,col+1] - im[row,col])/0.5
                        
                # If this is the rightmost pixel of this row -> don't trust the fit on the right edge (it is extrapolated)
                elif col == im.shape[1]-1 or np.ma.is_masked(im[row, col+1]): 
                    xgradient[row,col] = im[row,col]- im[row,col-1] # (im[row,col] - im_xEdges[row,col])/0.5
                
                else:
                    xgradient[row,col] = im_xEdges[row,col+1] - im_xEdges[row,col]

        
    if np.ma.isMA(im):
        xgradient = np.ma.masked_where(im.mask, xgradient)
        
    return xgradient/res_x


def Q_map(FMap_rads: npt.NDArray[np.float_], FOV_PE: float, t_esp: float, resFMap_PE: float) -> npt.NDArray[np.float_]:
    """Computes the "Q factor" map that modulates the intensity of the current-induced magnetic fields
    NB: the PE direction is assumed to be the x direction - ensure the image is properly oriented!


    Parameters
    ----------
    FMap_rads : npt.NDArray[np.float_]
        Field map in rad/s
    FOV_PE : float
        Field-of-view of the EPI image along the PE direction
    t_esp : float
        EPI echo spacing
    resFMap_PE : float
        Resolution of the field map along the EPI's PE direction (in case a separately-acquired field map is used)

    Returns
    -------
    npt.NDArray[np.float_]
        "Q factor" map 
    """

    G_pe_radsm = float("nan")*np.ma.ones_like(FMap_rads)
        
    for slc in range(FMap_rads.shape[2]):
    
        G_pe_radsm[:,:,slc] = xGradient(FMap_rads[:,:,slc], resFMap_PE) # rad/(s.m)

    G_pe_term = (1/(2*np.pi)) * FOV_PE * t_esp * G_pe_radsm
    
    Q_map = 1 + G_pe_term 
        
    return Q_map


def calc_shift_and_Q_maps(Mags: npt.NDArray[np.float_], 
                          Phs: npt.NDArray[np.float_], 
                          Mags_path: str, 
                          DPhTE12_path: str, 
                          affine_3D: npt.NDArray[np.float_],
                          SNR_mask: npt.NDArray[np.float_],
                          PE_dir: str,
                          TEs: npt.NDArray[np.float_], 
                          ETL: int,
                          t_esp: float, 
                          FOV_PE: float) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    
    """ Given a time-series of magnitude and phase images, computes maps
    of average voxel displacements and intensity scaling factor Q

    Parameters
    ----------
    Mags : npt.NDArray[np.float_]
        Time-series of 3D magnitude images
    Phs : npt.NDArray[np.float_]
        Time-series of 3D phase images
    Mags_path : str
        Full path of the magnitude images' nifti file. Needed for the spatial unwrapping procedure
    DPhTE12_path : str
        Full path of the delta-phase(TE1->TE2) image's nifti file. Needed for the spatial unwrapping procedure
    affine_3D : npt.NDArray[np.float_]
        4x4 affine trasnformation for each slice
    SNR_mask : npt.NDArray[np.float_]
        Mask for excluding low-SNR regions on the field map
    PE_dir : str
        Phase-econding direction. Possible values: "AP", "PA", "RL", "LR"
    TEs : npt.NDArray[np.float_]
        First and second echo times (in s)
    ETL : int
        EPI echo train length
    t_esp : float
        EPI echo spacing (in s)
    FOV_PE : float
        Field-of-view along the PE direction (in m)

    Returns
    -------
    Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]
        Voxel shift map (undistorted space) and "Q-factor" map
    """

    assert PE_dir in ["AP", "PA", "RL", "LR"]
    rot90ks = {"AP": 2, "PA": 0, "RL": 1, "LR": -1}
    rot90k = rot90ks[PE_dir]

    DPhTE12_sUW = get_DPhTE12_sUW(Mags, Phs, affine_3D, Mags_path, DPhTE12_path)

    shift_map_dist = rad2shift(DPhTE12_sUW, TEs, ETL, t_esp)

    # Smooth the voxel-shift map:
    shift_map_dist = gaussian_filter(shift_map_dist, (1,1))

    # Mask out low-SNR regions:
    shift_map_dist_SNRm = np.ma.masked_where(SNR_mask==0, shift_map_dist)

    shift_map_dist_SNRm = np.rot90(shift_map_dist_SNRm, k=rot90k, axes=(0,1)) 

    shift_map_UNdist = undistort_shift_map(shift_map_dist_SNRm)

    FMap_rads = shift2rads(shift_map_UNdist, ETL, t_esp)

    resFMap_PE = FOV_PE / ETL # Resolution of the EPI image (and, therefore, of the derived field map)

    Q = Q_map(FMap_rads, FOV_PE, t_esp, resFMap_PE)

    return np.rot90(shift_map_UNdist, k=-rot90k, axes=(0,1)), np.rot90(Q, k=-rot90k, axes=(0,1))


def shift_pixels(im: npt.NDArray[np.float_], shift_map: npt.NDArray[np.float_], PE_dir: str) -> npt.NDArray[np.float_]:
    """Shifts the voxels of 'im' according to the values in 'shift_map'

    Parameters
    ----------
    im : npt.NDArray[np.float_]
        Image whose voxels will be shifted
    shift_map : npt.NDArray[np.float_]
        Voxel-shift map
    PE_dir : str
        Phase-econding direction (i.e., direction along which shifting will be performed). Possible values: "AP", "PA", "RL", "LR"

    Returns
    -------
    npt.NDArray[np.float_]
        Input image after voxel shifting
    """

    assert im.ndim == 3 and im.shape == shift_map.shape
    
    assert PE_dir in ["AP", "PA", "RL", "LR"]
    rot90ks = {"AP": 2, "PA": 0, "RL": 1, "LR": -1}
    rot90k = rot90ks[PE_dir]

    im = np.rot90(im, k=rot90k, axes=(0,1)) 
    shift_map = np.rot90(shift_map, k=rot90k, axes=(0,1)) 

    im_shifted = np.ma.zeros_like(im)
    
    for slc in range(im_shifted.shape[2]):
            
        # TPS interpolation of 'im'
        xGrid, yGrid = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
        
        if np.ma.isMA(im): xGrid_train, yGrid_train = [ np.ma.masked_where(im[:,:,slc].mask, grid).compressed() for grid in [xGrid, yGrid] ]
        else:              xGrid_train, yGrid_train = [ xGrid.flatten(), yGrid.flatten() ]
    
        coords_train = np.stack([xGrid_train, yGrid_train], axis=-1)
                        
        kName = 'thin_plate_spline'; neigh = 15; s = 0 # 0.1
        interpolator = RBF(coords_train, im[:,:,slc].compressed() if np.ma.isMA(im) else im[:,:,slc].flatten(), kernel=kName, neighbors=neigh, smoothing=s, degree=1)
    
        # Evaluate the interpolated image at the coordinates that map to the new grid
        
        # Coords in the new grid
        xGrid, yGrid = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
       
        # Corresponding coords in the original img grid
        x_originalImGrid = xGrid + shift_map[:,:,slc].filled(0)
        
        im_shifted[:,:,slc] = interpolator(np.stack([x_originalImGrid.flatten(), yGrid.flatten()], axis=-1)).reshape(im[:,:,slc].shape)
    
    im_shifted = np.ma.masked_where(shift_map.mask, im_shifted)

    return np.rot90(im_shifted, k=-rot90k, axes=(0,1))


def distortion_correction(dist_im: npt.NDArray[np.float_],
                          Mags: npt.NDArray[np.float_], 
                          Phs: npt.NDArray[np.float_], 
                          Mags_path: str, 
                          DPhTE12_path: str, 
                          affine_3D: npt.NDArray[np.float_],
                          SNR_mask: npt.NDArray[np.float_],
                          gradTh: float,
                          PE_dir: str,
                          TEs: npt.NDArray[np.float_], 
                          ETL: int,
                          t_esp: float, 
                          FOV_PE: float,
                          gammaH: float) -> npt.NDArray[np.float_]:
    """
    Corrects the input image for voxel displacements and intensity modulation

    Parameters
    ----------
    dist_im : npt.NDArray[np.float_]
        Image to correct
    Mags : npt.NDArray[np.float_]
        Time-series of the magnitude images associated with 'dist_im'
    Phs : npt.NDArray[np.float_]
        Time-series of the phase images associated with 'dist_im'
    Mags_path : str
        Full path of the magnitude images' nifti file. Needed for the spatial unwrapping procedure
    DPhTE12_path : str
        Full path of the delta-phase(TE1->TE2) image's nifti file. Needed for the spatial unwrapping procedure
    affine_3D : npt.NDArray[np.float_]
        4x4 affine trasnformation for each slice
    SNR_mask : npt.NDArray[np.float_]
        Mask for excluding low-SNR regions on the field map
    gradTh : float
        Value in T/m for thresholding the gradient of the field map along the PE direction 
    PE_dir : str
        Phase-econding direction. Possible values: "AP", "PA", "RL", "LR"
    TEs : npt.NDArray[np.float_]
        First and second echo times (in s)
    ETL : int
        EPI echo train length
    t_esp : float
        EPI echo spacing (in s)
    FOV_PE : float
        Field-of-view along the PE direction (in m)
    gammaH : float
        Gyromagnetic ratio of the proton

    Returns
    -------
    npt.NDArray[np.float_]
        Image corrected for geometric distortion and masked based on a combination 
        of SNR maps and gradient of the field map along the PE direction
    """

    shift_map_UNdist, Q_map = calc_shift_and_Q_maps(Mags, Phs, Mags_path, DPhTE12_path, affine_3D, SNR_mask, PE_dir, TEs, ETL, t_esp, FOV_PE)

    DC_im = shift_pixels(dist_im, shift_map_UNdist, PE_dir)

    DC_im = DC_im * Q_map

    # Gradient mask:
    Q_pe_term = Q_map - 1

    grad_pe_Tm = Qpe_term_2_GpeTm(Q_pe_term, t_esp, FOV_PE, gammaH)

    grad_mask = gradient_mask(grad_pe_Tm, gradTh)

    DC_gradmasked_im = mask_array(DC_im, grad_mask)


    return DC_gradmasked_im



def gradient_mask(gradient_im: npt.NDArray[np.float_], gradTh: float) -> npt.NDArray[np.float_]:
    """Creates a binary mask by thresholding the gradient of the field map

    Parameters
    ----------
    gradient_im : npt.NDArray[np.float_]
        Gradient of the field map along the PE direction
    gradTh : float
        Gradient threshold

    Returns
    -------
    npt.NDArray[np.float_]
       Binary mask
    """

    grad_mask = abs(gradient_im) < gradTh  
    
    grad_mask = smooth_mask_edges( grad_mask.filled(0) )
    
    for slc in range(grad_mask.shape[2]):
        grad_mask[..., slc] = area_opening(grad_mask[..., slc], area_threshold=25)  

    return grad_mask


def rad2shift(DPh: npt.NDArray[np.float_], TEs: npt.NDArray[np.float_], ETL: int, t_esp: float) -> npt.NDArray[np.float_]:
    """
    Computes the map of voxel displacements given the phase accumulated 
    between the 2 echo times and some acquisition parameters

    Parameters
    ----------
    DPh : npt.NDArray[np.float_]
        Phase accumulated between the 2 echo times (in rad)
    TEs : npt.NDArray[np.float_]
        Echo times
    ETL : int
        EPI echo train length
    t_esp : float
        EPI echo spacing

    Returns
    -------
    npt.NDArray[np.float_]
        Map of voxel displacements
    """

    dTE = TEs[1] - TEs[0]
    
    FMap_rads = DPh / dTE
    
    shifts = ( FMap_rads/(2*np.pi) ) * ETL * t_esp
    # dy/res_y = dB/Gy_imaging

    return shifts


def rads2shift(FMap_rads, ETL, t_esp):
    
    shift_map = ( FMap_rads/(2*np.pi) ) * ETL * t_esp
    
    return shift_map


def shift2rads(shift_map, ETL, t_esp):
    
    # shift_map = ( FMap_rads/(2*np.pi) ) * ETL * t_esp     
    
    FMap_rads = 2*np.pi * shift_map / (ETL * t_esp)

    return FMap_rads


def shift2T(shift_map, ETL, t_esp, gamma_H):
    
    FMap_rads = shift2rads(shift_map, ETL, t_esp)

    # omega = gamma * B
    FMap_T = FMap_rads / gamma_H

    return FMap_T


def Qpe_term_2_GpeTm(Q_pe_term, t_esp, FOV_PE, gammaH):
    
    G_pe_radsm = (2*np.pi * Q_pe_term) / (FOV_PE * t_esp)
    
    G_pe_Tm = G_pe_radsm / gammaH
    
    return G_pe_Tm


