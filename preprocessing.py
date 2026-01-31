from simnibs.segmentation.samseg import gems
from simnibs.mesh_tools import mesh_io
from simnibs import transformations

import os
import shutil
import numpy as np
import numpy.typing as npt
from astropy.convolution import convolve, Gaussian2DKernel
from scipy import ndimage
from utils import mask_array, load_nii, save_nii, _affine_per_slice
import nibabel as nib
import re
from roipoly import RoiPoly
from matplotlib import pyplot as plt

# ---------------- Image/mesh registration ----------------

def _reg_nii2nii(fixedNii_Path, movingNii_Path, regNii_Path):
    """ Registers 2 images using a rigid transformation (6 degrees of freedom). 
    Saves the registered image and the transformation matrix

    Parameters
    ----------
    fixedNii_Path : str
        Full path of the Nifti file that will keep its initial position
    movingNii_Path : str
        Full path of the Nifti file to be registered
    regNii_Path : str
        Full path of the registered Nifti file

    Returns
    -------
    trans_mat : numpy array
        4x4 transformation matrix (in mm-to-mm)
    """

    out_dir_name = os.path.split(regNii_Path)
    out_dir = out_dir_name[0]  # Output directory
    out_filename = out_dir_name[1].split(".")[0]
    matPath = os.path.join(out_dir, out_filename + '_dof6.dat')

    RAS2LPS = np.diag([-1, -1, 1, 1])

    # Apply rigid Transformation (dof = 6)
    reg = gems.KvlRigidRegistration()
    reg.read_images(fixedNii_Path, movingNii_Path)
    reg.initialize_transform()
    reg.register()
    trans_mat = RAS2LPS@reg.get_transformation_matrix()@RAS2LPS

    reg.write_out_result(regNii_Path)  # Save registered image
    np.savetxt(matPath, trans_mat)    # Save transformation matrix

    return trans_mat


def _applyTrasnf2mesh(trans_mat, movingMesh_Path, regMesh_Path, refNifti_Path):
    """  Applies a given transformation to a mesh.
    Creates a label image from the reoriented mesh as a quality control of the transformation

    Parameters
    ----------
    trans_mat : numpy array
        4x4 transformation matrix (in mm-to-mm)
    movingMesh_Path : str
        Full path of the mesh to be moved
    regMesh_Path : str
        Full path of the moved mesh
    refNifti_Path : str
        Full path of a reference nifti file
    """

    mesh = mesh_io.read_msh(movingMesh_Path)
    node_coords = np.vstack((mesh.nodes[:].T, np.ones((1, mesh.nodes.nr))))
    node_coords = np.matmul(trans_mat, node_coords)[0:3]
    mesh.nodes.node_coord = node_coords.T
    mesh_io.write_msh(mesh, regMesh_Path)

    # Control label image
    ctrlNifti_Path = regMesh_Path.split(".msh")[0] + "_ctrl.nii.gz"
    transformations.interpolate_to_volume(
        mesh, refNifti_Path, ctrlNifti_Path, create_label=True)


def reg_mesh2nii(movingMesh_Path, movingNii_Path, fixedNii_Path, regMesh_Path):
    """ Reorients a mesh to fit to a given ("fixed") nifti image. Starts by registering the nifti image
    associated with the mesh to the "fixed" image, and then applies the obtained transformation to the mesh.

    Parameters
    ----------
    movingMesh_Path : str
        Full path of the mesh to be registered to the "fixed" nifti image
    movingNii_Path : str
        Full path of the nifti image associated with the mesh. It will also be registered to the "fixed" nifti image
    fixedNii_Path : str
        Full path of the "fixed" nifti image
    regMesh_Path : _type_
        Full path of the registered/reoriented mesh
    """

    out_dir = os.path.split(regMesh_Path)[0]

    movingNii_FN = os.path.split(movingNii_Path)[1].split(".")[
        0]  # without file extention
    # with file extention
    fixedNii_FN = os.path.split(fixedNii_Path)[1]

    regNii_Path = os.path.join(out_dir, movingNii_FN + "_to_" + fixedNii_FN)

    trans_mat = _reg_nii2nii(fixedNii_Path, movingNii_Path, regNii_Path)

    _applyTrasnf2mesh(np.linalg.inv(trans_mat),
                      movingMesh_Path, regMesh_Path, fixedNii_Path)


# -------- tSNR --------

def tSNR(tSeries):
    """ Computes the temporal SNR of a time-series of images

    Parameters
    ----------
    tSeries : ndarray
        t-series of images. Last dimension = time 

    Returns
    -------
    tSNR : ndarray
        Temporal SNR map
    """

    mean = np.mean(tSeries, axis=-1)
    std = np.std(tSeries, axis=-1)

    # Initialize the output:
    tSNR = np.zeros_like(mean)  # It will be changed by np.divide

    np.divide(mean, std, out=tSNR, where=std != 0)

    return tSNR


def gaussian_filter(im, sigma):
    """Low-pass filters each slice by convolving with a 2D Gaussian kernel

    Parameters
    ----------
    im : ndarray
        3D image to be smoothed (slice by slice)
    sigma : tuple
        Standard deviation of the Gaussian kernel in each dimension

    Returns
    -------
    ndarray
        Low-pass filtered image
    """
    assert im.ndim == 3  # (row, col, slc)

    kernel = Gaussian2DKernel(x_stddev=sigma[0], y_stddev=sigma[1])

    if np.ma.isMaskedArray(im):
        im_filt = np.ma.zeros_like(im)
    else:
        im_filt = np.zeros_like(im)

    for slc in range(im.shape[2]):
        im_filt[:, :, slc] = convolve(im[..., slc], kernel)

    if np.ma.isMaskedArray(im):
        im_filt = np.ma.masked_where(im.mask, im_filt)

    return im_filt


def smooth_mask_edges(mask, kernel=np.ones([3, 3])):

    assert mask.ndim == 3  # (row, col, slc)

    mask_s = np.zeros_like(mask)

    for slc in range(mask_s.shape[2]):

        mask_s[:, :, slc] = ndimage.binary_erosion(
            mask[:, :, slc],  structure=kernel)
        mask_s[:, :, slc] = ndimage.binary_dilation(
            mask_s[:, :, slc], structure=kernel)

    return mask_s


def SNR_map_to_mask(SNR_map: npt.NDArray[np.float_], th: float, sigma=(1, 1)):
    """ Creates a binary mask from a SNR map. First, low-pass filers each slice of
    the SNR map by convolving with a 2D Gaussian kernel. Then, thresholds it to get an
    initial  binary mask. Finally, smooths the edges of that mask

    Parameters
    ----------
    SNR_map : npt.NDArray[np.float_]
        SNR map
    th : float
        SNR map threshold. The mask will be 1 where SNR_map > th, and 0 where SNR_map <= th
    sigma : tuple, optional
        Standard deviation of the Gaussian kernel in each dimension, by default (1,1)

    Returns
    -------
    ndarray
        Binary mask
    """
    # Smooth the temporal SNR map
    SNR_map_LP = gaussian_filter(SNR_map, sigma)

    # Threshold the temporal SNR map:
    SNR_mask = SNR_map_LP > th

    SNR_mask = (np.ma.masked_where(
        SNR_mask == False, SNR_mask).astype(int)).filled(0)

    # Smooth the edges of the SNR mask:
    SNR_mask = smooth_mask_edges(SNR_mask)

    return SNR_mask


def create_SNR_mask(imgs: npt.NDArray[np.float_], th: float, SNR_map_path: str = "") -> npt.NDArray[np.float_]:
    """
    Creates a binary mask by thresholding the SNR map of a time series of images

    Parameters
    ----------
    imgs : npt.NDArray[np.float_]
        Time series of images
    th : float
        SNR map threshold. The mask will be 1 where SNR_map > th, and 0 where SNR_map <= th
    SNR_map_path : str, optional
        Path to save the SNR map, if desired

    Returns
    -------
    npt.NDArray[np.float_]
        Binary mask
    """

    SNR_map = tSNR(imgs)

    if SNR_map_path != "":
        np.save(SNR_map_path, SNR_map.filled("nan"))

    SNR_mask = SNR_map_to_mask(SNR_map, th)

    return SNR_mask




def separate_SMS(SMSnii_path: str, affine_3D: npt.NDArray[np.float_]):
    """
    Separates the SMS nifti file into single-slice nifti files

    Parameters
    ----------
    SMSnii_path : str
        Absolute path of the SMS nifti file
    affine_3D: npt.NDArray[np.float_]
        4x4 affine transformation for each slice
    """

    imgs_all_slcs = np.array(nib.load(SMSnii_path).get_fdata())

    N_slices = imgs_all_slcs.shape[2]

    if N_slices > 1:

        SSs_dir = SMSnii_path.split(".nii")[0] + "_Slcs/"

        os.makedirs(SSs_dir, exist_ok=1)

        SMS_FN = os.path.split(SMSnii_path)[1]
        SS_FNs = [SMS_FN.replace(".nii", "_Slc%d.nii.gz" % slc_i)
                    for slc_i in range(N_slices)]

        for slc in range(N_slices):
            save_nii(SSs_dir + SS_FNs[slc], imgs_all_slcs[:, :, slc, None, ...], affine_3D[:, :, slc])


def _avg_nii_at_TEk(tSeries_fPath, k, N_echoes, avg_fPath):
    """
    Parameters
    ----------
    tSeries_fPath : str
        Full path of the t-series of imgs to avg.
    k : int
        Which echo (after each excitation) to consider. k=1 -> avg imgs at TE1, etc
    N_echoes : int
        Number of echoe per excitation pulse.
    avg_fPath : str
        Full path to the average img

    """

    if not os.path.exists(avg_fPath):

        tSeries, affine = load_nii(tSeries_fPath)[0:2]

        avg_img = np.mean(tSeries[..., k-1::N_echoes], axis=-1)

        nib.save(nib.Nifti1Image(avg_img, affine), avg_fPath)


def manualMask(toMask_fPath, mask_fPath, masked_fPath=None, overW=0):
    """
    Parameters
    ----------
    toMask_fPath : str
        Full path to the magnitude image which will be used to create the mask.
    mask_fPath : str
        Full path to the mask which will be created or loaded.
    masked_fPath : str, optional
        Full path to the masked image (only needed if we want to save it)

    Returns
    -------
    mask : 3D array
        Created/loaded binary mask
    masked : 3D array
        Masked version of the input
    """

    file_ext = ".nii" if toMask_fPath.endswith(".nii") else ".npy"

    if file_ext == ".nii":
        nii = nib.load(toMask_fPath)
        toMask, affine = np.array(nii.get_fdata()), nii.affine
    else:
        toMask = np.load(toMask_fPath)

    if not os.path.exists(mask_fPath) or overW:
    
        if toMask.ndim == 4:
            toMask_3D = toMask[...,0] # Keep only the first time point for plotting purposes
        else:
            toMask_3D = toMask
    		
        plt.figure(figsize=np.array((16, 9))*1.3)
        plt.imshow(toMask_3D, cmap="gray")
        roi = RoiPoly(color='r')  # draw new ROI in red color

        mask = roi.get_mask(toMask_3D.squeeze()).astype(float)
        if mask.ndim == 2:
            mask = mask[..., np.newaxis]

        nib.save(nib.Nifti1Image(mask, affine),
                 mask_fPath) if file_ext == ".nii" else np.save(mask_fPath, mask)

        masked = mask_array(toMask, mask)

        if masked_fPath != None:
            nib.save(nib.Nifti1Image(masked.filled(0), affine), masked_fPath) if file_ext == ".nii" else np.save(
                masked_fPath, masked.filled(float("nan")))

    else:

        mask = np.array(nib.load(mask_fPath).get_fdata()
                        ) if file_ext == ".nii" else np.load(mask_fPath)

        masked = mask_array(toMask, mask)

    return mask, masked


def _create_mask(Mag_fPath, masks_dir, fVal=0.5, gVal=0, manual=0, eKernel=3, verbose=1, Ph_fPath=""):

    # -------------------------------------------------------------------------
    def loadAndPlot(Mag_fPath, mask_fPath, figsize=np.array([2,1])*3.7, Ph_fPath=""):

        mask  = np.ma.masked_values(load_nii(mask_fPath)[0], 0)
        mag   = load_nii(Mag_fPath)[0]
        N_col = 2

        vmax_mag = np.quantile(mag, .98)

        if len(Ph_fPath) > 0:
            N_col = 4
            figsize=np.array([4,1])*3.7
            ph = load_nii(Ph_fPath)[0]
            vmax_ph = np.quantile(abs(ph), .98)

        fig, axes = plt.subplots(1, N_col, sharex="all", sharey="all", figsize=figsize)
        fig.suptitle(mask_fPath.split("/")[-1].split(".nii")[0])

        ax = axes[0]
        ax.imshow(np.rot90(mag), vmin=0, vmax=vmax_mag, cmap="gray")
        ax.imshow(np.rot90(mask), vmin=0, vmax=1.2, cmap="Greens", alpha=0.4)
        
        ax = axes[1]
        ax.imshow(np.rot90(np.ma.masked_where(mask==0, mag)), vmin=0, vmax=vmax_mag, cmap="gray")
        
        if len(Ph_fPath) > 0: 

            ax = axes[2]
            ax.imshow(np.rot90(ph), vmin=-vmax_ph, vmax=vmax_ph, cmap="bwr")
            ax.imshow(np.rot90(mask), vmin=0, vmax=1.2, cmap="Greens", alpha=0.4)

            ax = axes[3]
            ax.imshow(np.rot90(np.ma.masked_where(mask==0, ph)), vmin=-vmax_ph, vmax=vmax_ph, cmap="bwr")

        [ax.set_axis_off() for ax in axes]

        plt.tight_layout(); plt.show()
    # -------------------------------------------------------------------------

    Mag_FN = Mag_fPath.split("/")[-1]

    if not os.path.isdir(masks_dir):
        os.mkdir(masks_dir)

    Mask_FN = [FN for FN in os.listdir(
        masks_dir) if Mag_FN[:-4] in FN and "Mask" in FN]  # print(Mask_FN)

    # 1.1. If a mask exists
    if len(Mask_FN) != 0:

        assert len(Mask_FN) == 1, "Warning: more than 1 mask found: " + str(Mask_FN)

        Mask_FN = Mask_FN[0]
        mask_fPath = masks_dir + Mask_FN

        if verbose:
            ans = input(
                "A mask already exists: %s\nEnter - Keep it   1 - Recompute it   2 - Plot it\nans: " % Mask_FN)
            while ans not in ["", "1", "2"]:
                ans = input(
                    "Invalid input: %s\nEnter - Keep it   1 - Recompute it   2 - Plot it\nans: " % ans)
        else:
            ans = ""

        if ans == "2":  # Plot the existing mask

            loadAndPlot(Mag_fPath, mask_fPath, Ph_fPath=Ph_fPath)

            ans = input("Enter - Keep it   1 - Recompute it\nans:")
            while ans not in ["", "1"]:
                ans = input(
                    "Invalid input: %s\nEnter - Keep it   1 - Recompute it\nans: " % ans)

        # If the mask is to be kept:
        if ans == "":
            mask = load_nii(mask_fPath)[0]

    # 1.3. If a mask does not exist / is to be recomputed:
    if len(Mask_FN) == 0 or ans == "1":

        ans = "1"

        while ans != "":

            if len(Mask_FN) > 0:  # If a mask existed, delete it

                if isinstance(Mask_FN, list):
                    Mask_FN = Mask_FN[0]
                os.remove(masks_dir + Mask_FN)

            f_search = re.search(r"f=?([0-9]*\.?[0-9]+)", ans)
            if f_search:
                fVal = float(f_search.group(1))

            g_search = re.search(r"g=?([0-9]*\.?[0-9]+)", ans)
            if g_search:
                gVal = float(g_search.group(1))

            eKernel_search = re.search("eKernel=?([0-9])", ans)
            if eKernel_search:
                eKernel = eKernel_search.group(1)

            Mask_FN = Mag_FN.split(".nii")[0] + "_Mask_f" + str(fVal).replace(
                ".", "") + "_g" + str(gVal).replace(".", "") + ".nii"
            mask_fPath = masks_dir + Mask_FN
            BE_fPath = masks_dir + Mag_FN.split(".nii")[0] + "_brain.nii.gz"

            # bet mask
            # https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide

            print("\nRunning bet2 on " + Mag_FN + " with f = " +
                    str(fVal) + " and g = " + str(gVal))
            cmd = '. /mnt/depot64/fsl/fsl.6.0.4/etc/fslconf/fsl.sh; /mnt/depot64/fsl/fsl.6.0.4/bin/bet2 ' + Mag_fPath + ' ' + BE_fPath + \
                ' -f ' + str(fVal) + ' -g ' + str(gVal)  # + ' -m'
            os.system(cmd)

            ''' Further processing of the bet mask '''

            # Load the brain extracted (BE) mag image just created & add the slice dimension:
            BE_Mag, affine, _ = load_nii(BE_fPath)

            if BE_Mag.ndim == 2:
                BE_Mag = BE_Mag[:, :, np.newaxis]

            # Create a binary mask from the BE mag image:
            BE_binary = ndimage.binary_erosion(BE_Mag, structure=np.ones(
                (eKernel, eKernel, 1))).astype(BE_Mag.dtype)

            # Shrink slightly the mask:
            mask_dilated = ndimage.binary_dilation(BE_binary,  structure=np.ones(
                (eKernel, eKernel, 1))).astype(BE_binary.dtype)
            mask_eroded = ndimage.binary_erosion(mask_dilated, structure=np.ones(
                (eKernel, eKernel, 1))).astype(BE_binary.dtype)
            mask = mask_eroded

            save_nii(mask_fPath, mask.astype(np.int8),  affine, descrip=f"bet mask with f={fVal} and g={gVal}")

            loadAndPlot(Mag_fPath, mask_fPath, Ph_fPath=Ph_fPath)

            ans = input(
                "Enter -> keep it\nf? -> bet mask with f=? (f in [0,1]; default: f=0.5)\ng? -> bet mask with g=?\nans: ")
            while ans not in ["", "m"] and not re.search(r"f=?([0-9]*\.?[0-9]+)", ans) and not re.search(r"g=?([0-9]*\.?[0-9]+)", ans):
                ans = input(
                    "Invalid input: %s\nEnter -> keep it\nf? -> bet mask with f=? (f in [0,1]; default: f=0.5)\ng? -> bet mask with g=?\nans: " % ans)

    return mask, fVal, gVal


def masking_pipeline(Mags_path: str, affine_3D: npt.NDArray[np.float_], N_echoes: int, masks_dir: str) -> npt.NDArray[np.int_]:
    """Performs all the steps involved in the creation of a brain mask from a time-series of magnitude images:
        1. If SMS acquisition, separates the SMS nifti into single-slice niftis
        2. Averages the t-series of magnitude images at the first echo
        3. Creates a mask for each slice using BET or manual drawing
        4. If SMS acquisition, gathers all single-slice masks into a multi-slice mask

    Parameters
    ----------
    Mags_path : str
        Full path of the time series of magnitude images
    affine_3D : npt.NDArray[np.float_]
        4x4 affine transformation for each slice
    N_echoes : int
        Number of echoes per excitation pulse
    masks_dir : str
        Directory where the final mask will be saved

    Returns
    -------
    npt.NDArray[np.int_]
        Created binary mask
    """

    N_slcs = affine_3D.shape[2]

    Mags_FN = os.path.split(Mags_path)[1]

    if N_slcs == 1:

        avg_path = os.path.join(
            masks_dir, Mags_FN.replace(".nii", "_avgTE1.nii"))
        _avg_nii_at_TEk(Mags_path, 1, N_echoes, avg_path)

        mask, _, _ = _create_mask(avg_path, masks_dir)

        os.remove(avg_path)
        if os.path.exists(avg_path.replace(".nii", "_brain.nii.gz")): os.remove(avg_path.replace(".nii", "_brain.nii.gz"))
    else:

        separate_SMS(Mags_path, affine_3D)

        SSs_dir = Mags_path.split(".nii")[0] + "_Slcs/"

        masks_slcs_l = []

        # Get a mask for each slice:
        for slc in range(N_slcs):

            Mags_slc_path = os.path.join(SSs_dir, Mags_FN.replace(".nii", "_Slc%d.nii.gz" % slc))
            Mags_slc_avg_path = os.path.join(SSs_dir, Mags_FN.replace(".nii", "_Slc%d_avgTE1.nii.gz" % slc))

            _avg_nii_at_TEk(Mags_slc_path, 1, N_echoes, Mags_slc_avg_path)

            mask_slc, _, _ = _create_mask(Mags_slc_avg_path, SSs_dir)

            masks_slcs_l.append(mask_slc[:, :, 0])

            os.remove(Mags_slc_path)
            if os.path.exists(Mags_slc_avg_path.replace(".nii", "_brain.nii.gz")): os.remove(Mags_slc_avg_path.replace(".nii", "_brain.nii.gz"))

        # Stack the single-slice masks and save the resulting 3D mask:

        mask = np.stack(masks_slcs_l, axis=2)

        SMS_mask_fPath = os.path.join(
            masks_dir, Mags_FN.replace(".nii", "_avgTE1_Mask.nii"))

        SMS_affine = nib.load(Mags_path).affine

        nib.save(nib.Nifti1Image(mask, SMS_affine), SMS_mask_fPath)

    return mask.astype(int)


def get_brain_mask(Mags_path: str, affine_3D: npt.NDArray[np.float_], N_echoes: int, masks_dir: str, recompute_mask: int = 0) -> npt.NDArray[np.int_]:
    """Runs the full masking pipeline if a mask does not exist or is to be recomputed

    Parameters
    ----------
    Mags_path : str
        Full path of the time series of magnitude images
    affine_3D : npt.NDArray[np.float_]
        4x4 affine transformation for each slice
    N_echoes : int
        Number of echoes per excitation pulse
    masks_dir : str
        Directory where the final mask will be saved
    recompute_mask : int, optional
        Whether or not to replace an existing mask

    Returns
    -------
    npt.NDArray[np.int_]
        Binary mask
    """

    Mags_FN = os.path.split(Mags_path)[1]

    Mask_FN = [FN for FN in os.listdir(masks_dir) if Mags_FN[:-4] in FN and "Mask" in FN]

    if len(Mask_FN) == 0 or recompute_mask:

        mask = masking_pipeline(Mags_path, affine_3D, N_echoes, masks_dir)

    else:
        assert len(Mask_FN) == 1
        mask = np.array(nib.load( os.path.join(masks_dir, Mask_FN[0]) ).get_fdata())

    return mask


def prepare4betMasking(Mags, Phs, affine, echo, N_echoes, avg_mags_path, ph_t0_path, slc_thickness=0):

        # Average the magnitude images at the echo-th echo
        avg_img = np.mean(Mags[..., echo::N_echoes], axis=-1)
        save_nii(avg_mags_path, avg_img, affine)

        # Save the phase image at the first repetition (also at the echo-th echo)
        save_nii(ph_t0_path, Phs[...,echo], affine)

        # Separate the slices for better bet performance (bad results otherwise because it assumes the slices are adjacent to each other)
        affine_3D = _affine_per_slice(affine, Mags.shape[2], slc_thickness=slc_thickness)

        separate_SMS(avg_mags_path, affine_3D)
        separate_SMS(ph_t0_path, affine_3D)


def run_bet_in_eachSlice_and_combineSlices(mag_path, ph_path, mask_path, temp_dir, mask_shape, mask_affine, fVals=0.5, gVals=0):

        mask_slcs = np.zeros(mask_shape, dtype=np.int8)
        fVal_slcs = np.zeros(mask_shape[2])
        gVal_slcs = np.zeros(mask_shape[2])

        if not isinstance(fVals, (np.ndarray, list, tuple)):
            fVals = [fVals]*mask_shape[2]
        if not isinstance(gVals, (np.ndarray, list, tuple)):
            gVals = [gVals]*mask_shape[2]

        for slc in range(mask_shape[2]):

            mag_fn = os.path.basename(mag_path).split(".nii")[0] 
            ph_fn  = os.path.basename(ph_path).split(".nii")[0]

            mag_slc_path  = mag_path.split(".nii")[0] + f"_Slcs/{mag_fn}_Slc{slc}.nii.gz"
            ph_slc_path   = ph_path.split(".nii")[0] + f"_Slcs/{ph_fn}_Slc{slc}.nii.gz"

            mask_slcs[:,:,slc,None], fVal_slcs[slc], gVal_slcs[slc] = \
            _create_mask(mag_slc_path, f"{temp_dir}/", fVal=fVals[slc], gVal=gVals[slc], manual=0, eKernel=3, verbose=1, Ph_fPath=ph_slc_path)

        descripton = "fVals = " + ",".join([str(f) for f in fVal_slcs]) + \
                        " ; gVals = " + ",".join([str(g) for g in gVal_slcs])

        save_nii(mask_path, mask_slcs, mask_affine, descrip=descripton)


def remove_temporary_masking_files(temp_dir, endings):
    
    FNs2remove = [fn for fn in os.listdir(temp_dir) if np.any([fn.endswith(ending) for ending in endings])]

    for FN in FNs2remove:

        if os.path.isdir(f"{temp_dir}/{FN}"):
            shutil.rmtree(f"{temp_dir}/{FN}")
        else:
            os.remove(f"{temp_dir}/{FN}")


def plot_masks_allSlcs(mag, ph, mask, suptitle=""):

    mag, ph, mask = [np.rot90(arr, axes=(0,1)) for arr in [mag, ph, mask]]
    mask = np.ma.masked_values(mask, 0)

    vmax_mag = np.quantile(mag, .98)
    vmax_ph = np.quantile(abs(ph), .98)

    N_col = mag.shape[2]
    N_row = 3
    figsize=np.array([N_col, N_row])*2.2

    fig, axes = plt.subplots(N_row, N_col, sharex="all", sharey="all", figsize=figsize)
    fig.suptitle(suptitle)

    for slc in range(N_col):

        ax = axes[0,slc]
        ax.imshow(mag[:,:,slc], vmin=0, vmax=vmax_mag, cmap="gray", interpolation="nearest")
        ax.imshow(mask[:,:,slc], vmin=0, vmax=1.2, cmap="Greens", alpha=0.4, interpolation="nearest")
        
        ax = axes[1,slc]
        ax.imshow(np.ma.masked_where(mask==0, mag)[:,:,slc], vmin=0, vmax=vmax_mag, cmap="gray", interpolation="nearest")
        
        ax = axes[2,slc]
        ax.imshow(np.ma.masked_where(mask==0, ph)[:,:,slc], vmin=-vmax_ph, vmax=vmax_ph, cmap="bwr", interpolation="nearest")

    [ax.set_axis_off() for ax in axes.flatten()]

    plt.tight_layout(); plt.show()


def _remove_single_voxel_artifacts(binary_mask, th_island, th_hole):
    """
    Removes single voxel artifacts from a binary mask.
    """

    labeled_mask1, _ = ndimage.label(binary_mask)
    component_sizes1 = np.bincount(labeled_mask1.ravel())
    large_components1 = component_sizes1 > th_island
    large_components1[0] = 0  # Background is not a component
    binary_mask_m = large_components1[labeled_mask1]

    binary_mask_m_neg = (~(binary_mask_m.astype(bool))).astype(int)
    labeled_mask_m2, _ = ndimage.label(binary_mask_m_neg)
    component_sizes2 = np.bincount(labeled_mask_m2.ravel())
    large_components2 = component_sizes2 > th_hole
    binary_mask_m_neg_m = large_components2[labeled_mask_m2]
    binary_mask_comb_m = np.logical_or(~binary_mask_m_neg_m,binary_mask_m)

    return binary_mask_comb_m


def remove_single_voxel_artifacts_multislice(binary_mask, th_island, th_hole):

    binary_mask_m = np.zeros_like(binary_mask)

    for slc in range(binary_mask.shape[2]):
        binary_mask_m[:,:,slc] = _remove_single_voxel_artifacts(binary_mask[:,:,slc], th_island=th_island, th_hole=th_hole)

    return binary_mask_m


def intersect_masks(array1, array2):

    masks_intersect = np.logical_and(~array1.mask, ~array2.mask)
    masks_intersect = remove_single_voxel_artifacts_multislice(masks_intersect,25,1)
    masks_intersect = smooth_mask_edges(masks_intersect,kernel=np.ones([3,3]))

    array1_m, array2_m = [np.ma.masked_where(masks_intersect==0, arr) for arr in [array1, array2]]

    return masks_intersect, array1_m, array2_m



def calc_variance_mask(img_series, variance_th, variance_sigmaGauss, affine, variance_mask_path="", variance_map_path=""):

    variance = np.ma.var(img_series, axis=3)  # (row, col, slc)
    
    if variance_sigmaGauss > 0:
        variance = gaussian_filter(variance, sigma=(variance_sigmaGauss,variance_sigmaGauss)) 

    bet_mask = ~img_series[:,:,:,0].mask

    variance_mask = np.logical_and((variance <= variance_th).filled(0), bet_mask) 
    variance_mask = remove_single_voxel_artifacts_multislice(variance_mask, th_island=25, th_hole=1)
    variance_mask = smooth_mask_edges(variance_mask, kernel=np.ones([3,3]))

    if variance_map_path != "":
        save_nii(variance_map_path, variance.filled(-np.inf).astype(np.float32), affine, descrip=f"variance_sigmaGauss={variance_sigmaGauss}")

    if variance_mask_path != "":
        save_nii(variance_mask_path, variance_mask.astype(np.int8), affine, descrip=f"variance_sigmaGauss={variance_sigmaGauss} , variance_th={variance_th}")
    
    return variance_mask


def save_first_time_point(in_path, out_path, echo):

    imgs, affine, _ = load_nii(in_path)

    img_echo = imgs[:,:,:,echo]

    save_nii(out_path, img_echo.astype(np.int16), affine)


def choose_echoes_and_downsample_in_time(imgs, echo, N_echoes, t_step):

    imgs_echo    = imgs[:,:,:,echo::N_echoes]
    imgs_echo_DS = imgs_echo[:,:,:,0::t_step]

    return imgs_echo_DS


def run_MCFLIRT(in_path: str, out_path: str, reffile_fPath: str = "", cost: str = "normcorr", verbose=0, remove_out_nii=0):
    """
    Runs FSL tool MCFLIRT to check for movement in a time-series of images. 
    If 'reffile_fPath' is not provided, images are registered to the 1st volume of the time series.
    A file named in_path_mcf.par is saved with the transformation parameters for each time
    point (rot_x, rot_y, rot_z [rad], trans_x, trans_y, trans_z [mm])

    Parameters
    ----------
    in_path : str
        Full path to the input time series of images
    out_path : str
        Full path to the motion-corrected time series of images
    reffile_fPath : str, optional
        Full path to a separate image to be used as the target for registration
    cost : str, optional
        Cost function for the linear registration, by default "normcorr"
    """
 
    costs = ["mutualinfo","woods","corratio","normcorr","normmi","leastsquares"]   
    assert cost in costs
    
    cmd = f"mcflirt -in {in_path} -out {out_path} -cost {cost} -dof 6 -plots -verbose {verbose}" #-report
    if reffile_fPath != "": cmd += f" -reffile {reffile_fPath}"
    else: cmd += " -refvol 0"
    
    os.system(cmd)

    if remove_out_nii: 
        os.remove(out_path.split(".nii")[0] + ".nii.gz")


def load_MCFLIRT_params_per_set(mcf_dir, set1_suffix, set2_suffix):
    
    FNs_in_mcf_dir = os.listdir(mcf_dir)

    sets_suffixes = [set1_suffix, set2_suffix]

    params_per_set_l   = []
    t_axis_per_set_l   = []
    t_vlines_per_set_l = []
    SNs_per_set_l      = []
    tags_per_set_l     = [] 

    for set_i in [0,1]:

        params_l   = []
        t_vlines_l = [0]
        tags_l     = []

        set_FNs = np.array([FN for FN in FNs_in_mcf_dir if FN.endswith(sets_suffixes[set_i])])
        set_SNs = np.array([int(FN.split("_")[0][1:]) for FN in set_FNs])

        set_FNs = set_FNs[np.argsort(set_SNs)]
        set_SNs = set_SNs[np.argsort(set_SNs)]
    
        for FN in set_FNs:
            # print(FN)

            mcf_params = np.loadtxt(f"{mcf_dir}/{FN}")
            if mcf_params.ndim == 1:
               mcf_params = mcf_params[None,:] 

            params_l.append(mcf_params)
            
            t_vlines_l.append(t_vlines_l[-1] + len(params_l[-1]))

            tags_l.append(FN.split("MB5_")[1].split("_set")[0])
        
        params = np.concatenate(params_l, axis=0) # (t, parameters)
        params_per_set_l.append(params_l)

        t_axis_per_set_l.append(np.arange(params.shape[0]))
        t_vlines_per_set_l.append(np.array(t_vlines_l))
        
        SNs_per_set_l.append(set_SNs)
        tags_per_set_l.append(tags_l)

    return params_per_set_l, t_axis_per_set_l, t_vlines_per_set_l, SNs_per_set_l, tags_per_set_l


def petra_vline_locations(cmrrSNs_perSet_l, petras_SNs):

    petras_SNs_and_indices_perSet_l = []

    for set_SNs in cmrrSNs_perSet_l:

        vlines_indices_l = []

        for petraSN in petras_SNs:

            vline_idx = np.argwhere(set_SNs > petraSN) 

            if len(vline_idx)>0: # If the last scan was not a petra
                vline_idx = vline_idx[0,0]
            else:
                vline_idx = -1

            vlines_indices_l.append(vline_idx)
        
        petras_SNs_and_indices_perSet_l += [(petras_SNs, np.array(vlines_indices_l))]

    return petras_SNs_and_indices_perSet_l


def _ax_vlines(ax, axvlines, petras_SNs_and_indices, ylims):
    
    dy = ylims[1] - ylims[0]

    for line_idx,x in enumerate(axvlines):
        
        if line_idx in petras_SNs_and_indices[1]:
            ax.axvline(x, c="orange", ls="--", alpha=0.5)
            SN = petras_SNs_and_indices[0][np.argwhere(petras_SNs_and_indices[1]==line_idx)[0,0]]
            ax.text(x, ylims[0]+0.05*dy, f"{SN}\n", fontsize=10, color='k', ha="center", va="bottom")
        else:
            ax.axvline(x, c="k", ls="--", alpha=0.5)

def _ax_text(ax, axvlines, text_l, ylims):
    
    centers = (axvlines[:-1] + axvlines[1:])/2
    dy = ylims[1] - ylims[0]
    
    for center, txt in zip(centers, text_l):
        ax.text(center, ylims[0]+0.05*dy, txt, fontsize=10, color='k', ha="center", va="bottom")
    

def plot_mcf_params_sep_sets(params_l, figsize=(10,5), t_axis=[], axvlines=[], epi_SNs=[], epi_tags=[], 
                             petras_SNs_and_indices=[], ylims_trans=None, ylims_rot=None, ylims_FWD=None, suptitle=""):
    
    labels = ["x","y","z"]
    colors = ["b", "r", "g"]

    if len(t_axis) == 0:
        t_axis = [np.arange(2,params_l[0].shape[0])]*2
    elif not isinstance(t_axis, list):
        t_axis = [t_axis]*2

    fig, axes = plt.subplots(3, len(params_l), sharex="col", sharey="row", figsize=figsize)
    fig.suptitle(suptitle)

    for col in range(len(params_l)):

        epi_texts = [f"{SN}\n{tag}" for SN,tag in zip(epi_SNs[col],epi_tags[col])]

        params = np.concatenate(params_l[col], axis=0) # (t, params)
        
        ax = axes[0,col] 
        ax.set_title(["Set 1", "Set 2"][col])
        ax.axhline(0, c="k", lw=0.75)
        for idx in range(3): ax.plot(t_axis[col], params[:,3+idx], c=colors[idx], label=f"trans {labels[idx]}")
        _ax_vlines(ax, axvlines, petras_SNs_and_indices[col], ylims_trans)
        _ax_text(ax, axvlines, epi_texts, ylims_trans)
        if col == 0: ax.legend(bbox_to_anchor=[1,.5], loc="center left"); 
        ax.set_ylim(ylims_trans);  
            
        ax = axes[1,col] 
        ax.axhline(0, c="k", lw=0.75)
        for idx in range(3): ax.plot(t_axis[col], np.rad2deg(params[:,idx]), c=colors[idx], label=f"rot {labels[idx]}")  
        _ax_vlines(ax, axvlines, petras_SNs_and_indices[col], ylims_rot)     
        _ax_text(ax, axvlines, epi_texts, ylims_rot)
        if col == 0: ax.legend(bbox_to_anchor=[1,.5], loc="center left"); 
        ax.set_ylim(ylims_rot);  
        
        ax = axes[2,col]
        ax.axhline(0, c="k", lw=0.75)
        frameWiseDisplacement = np.sum(50*abs(params[:,:3]), axis=1) + np.sum(abs(params[:,3:]), axis=1)
        ax.plot(t_axis[col], frameWiseDisplacement, lw=1, c="k")
        _ax_text(ax, axvlines, epi_texts, ylims_FWD)
        _ax_vlines(ax, axvlines, petras_SNs_and_indices[col], ylims_FWD)     
        ax.set_ylim(ylims_FWD);  

        ax.set_xlim([-2+t_axis[col][0],2+t_axis[col][-1]])

    axes[0,0].set_ylabel("transl [mm]"); axes[1,0].set_ylabel("rot [deg]"); axes[2,0].set_ylabel("frame-wise\ndisplacement [mm]")

    plt.tight_layout()
    
    return fig


def plot_mcf_params_and_fft(params, params_fft, FWD, FWD_fft, t_axis, f_axis, ylims_trans=None, ylims_rot=None, ylims_FWD=None, 
                            ylims_fft_trans=None, ylims_fft_rot=None, ylims_fft_FWD=None, fft_vlines=[], figsize=(10,5), suptitle=""):
    
    labels = ["x","y","z"]
    colors = ["b", "r", "g"]
    lw = 1
    f_range = f_axis[-1] - f_axis[0]

    fig, axes = plt.subplots(3,2, sharex="col", figsize=figsize)
    fig.suptitle(suptitle, fontsize=11)

    ax = axes[0,0]; ax.set_title("time series", fontsize=11)
    for idx in range(3): 
        ax.plot(t_axis, params[:,3+idx], c=colors[idx], lw=lw, label=f"trans {labels[idx]}", alpha=1-0.15*idx)
    ax.legend(bbox_to_anchor=[1,.5], loc="center left")
    ax.set_ylim(ylims_trans)
    ax.set_ylabel("transl [mm]"); ax.grid(axis="y")
        
    ax = axes[1,0] 
    for idx in range(3): 
        ax.plot(t_axis, np.rad2deg(params[:,idx]), c=colors[idx], lw=lw, label=f"rot {labels[idx]}", alpha=1-0.15*idx)  
    ax.legend(bbox_to_anchor=[1,.5], loc="center left"); 
    ax.set_ylim(ylims_rot);  
    ax.set_ylabel("rot [deg]"); ax.grid(axis="y")

    ax = axes[2,0]
    ax.plot(t_axis, FWD, lw=1, c="k")   
    ax.set_ylim(ylims_FWD)
    ax.set_ylabel("Frame-Wise\nDisplacement [mm]"); ax.grid(axis="y")

    ax.set_xlim(t_axis[[0,-1]]); ax.set_xlabel("time [s]")

    ax = axes[0,1]; ax.set_title("|FFT(time series)|", fontsize=11)
    for idx in range(3): 
        ax.plot(f_axis, params_fft[:,3+idx], c=colors[idx], lw=lw, alpha=1-0.15*idx)
        ax.text(0.04*f_range, (0.9-0.1*idx)*ylims_fft_trans[1], f"max = {np.amax(params_fft[:,3+idx]):.2f}", fontsize=10, color=colors[idx], ha="left", va="top")
    ax.set_ylim(ylims_fft_trans); ax.grid(axis="x")
    ax.set_ylabel("|FFT(transl)|") 

    ax = axes[1,1] 
    for idx in range(3): 
        ax.plot(f_axis, params_fft[:,idx], c=colors[idx], lw=lw, alpha=1-0.15*idx)
        ax.text(0.04*f_range, (0.9-0.1*idx)*ylims_fft_rot[1], f"max = {np.amax(params_fft[:,idx]):.2f}", fontsize=10, color=colors[idx], ha="left", va="top")
    ax.set_ylim(ylims_fft_rot); ax.grid(axis="x")
    ax.set_ylabel("|FFT(rot)|")

    ax = axes[2,1] 
    ax.plot(f_axis, FWD_fft, c="k", lw=lw, alpha=1)  
    ax.set_ylim(ylims_fft_FWD); ax.grid(axis="x")
    ax.set_ylabel("|FFT(FWD)|")
    ax.text(0.04*f_range, 0.9*ylims_fft_FWD[1], f"max = {np.amax(FWD_fft):.2f}", fontsize=10, color='k', ha="left", va="top")

    ax.set_xlim(1.01*f_axis[[0,-1]]); ax.set_xlabel("freq [Hz]")

    for line in fft_vlines:
        axes[0,1].axvline(line, c="k", ls="--", alpha=0.5)
        axes[1,1].axvline(line, c="k", ls="--", alpha=0.5)
        axes[2,1].axvline(line, c="k", ls="--", alpha=0.5)

    plt.tight_layout()
    # plt.show()
    
    return fig


def plot_MCFLIRT_motion_params(params, figsize=(10,5), t_axis=[], axvlines=[], ylims_trans=None, ylims_rot=None):
    
    labels = ["x","y","z"]
    colors = ["b", "r", "g"]

    if len(t_axis) == 0:
        t_axis = np.arange(params.shape[0])

    plt.figure(figsize=figsize)

    plt.subplot(2,1,1)
    for idx in range(3):
        plt.plot(t_axis, params[:,3+idx], c=colors[idx], label=f"trans {labels[idx]}")
    for x in axvlines:
        plt.axvline(x, c="k", ls="--", alpha=0.5)
         
    plt.legend(); plt.xlim([t_axis[0],t_axis[-1]]); plt.ylim(ylims_trans); plt.ylabel("transl [mm]")

    plt.subplot(2,1,2)
    for idx in range(3):
        plt.plot(t_axis, params[:,idx], c=colors[idx], label=f"rot {labels[idx]}")  
    for x in axvlines:
        plt.axvline(x, c="k", ls="--", alpha=0.5)      
    plt.legend(); plt.xlim([t_axis[0],t_axis[-1]]); plt.ylim(ylims_rot); plt.xlabel("t"); plt.ylabel("rot [rad]")

    plt.tight_layout()
    plt.show()