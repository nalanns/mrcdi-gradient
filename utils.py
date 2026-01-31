import nibabel as nib
from dicom_parser import Image as dcmImage
import numpy as np
import numpy.typing as npt
import os
import re
from matplotlib import pyplot as plt, animation as animation
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import warnings; 
from numpy.fft import fft, fftshift
from scipy.ndimage import binary_dilation


def load_nii(fullPath: str):
    """
    Parameters
    ----------
    fullPath : str
        Full path to the Nifti file to be loaded
    Returns
    -------
    img : ndarray
        3D image or time-series of 3D images
    affine : ndarray
        4x4 affine matrix
    header : nifti header 
    """

    nii = nib.load(fullPath)
    img, affine, header = np.array(nii.get_fdata()), nii.affine, nii.header

    # Ensure the slice dimension exists
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    return img, affine, header

def save_nii(full_path: str, im: npt.NDArray[np.float_], affine: npt.NDArray[np.float_], descrip: str = ""):

    nii_img = nib.Nifti1Image(im, affine)
    nii_img.header["descrip"] = descrip

    nib.save(nii_img, full_path)


def apply_recursive(input_array, output_array, dim_subarrays, func, *args, **kwargs):
    """
    Recursively applies a function to subarrays of an dim_subarrays-dimensional array.
    
    Parameters:
    - input_array: The input array to process.
    - output_array: The output array where results will be stored.
    - func: The function to apply to dim_subarrays-dimensional subarrays.
    - *args: Additional arguments for `func`.
    
    Returns:
    - Modified output_array.
    """

    # Base case:
    if input_array.ndim == dim_subarrays:
        # Apply the function to the dim_subarrays-dimensional subarray
        output_array[...] = func(input_array, *args, **kwargs) # In-Place Modification - it modifies the existing array in place, thus preserving the object reference and memory location
    
    # Recursive case:
    # Recursively apply the function to subarrays along the last axis
    else:
        for idx_in_last_dim in range(input_array.shape[-1]):
            apply_recursive(input_array[...,idx_in_last_dim], output_array[...,idx_in_last_dim], dim_subarrays, func, *args, **kwargs)

    return output_array


def mask_array(array, mask):
    """
    The elements of **array** where **mask** = 0 will be discarded

    Parameters
    ----------
    array : ndarray
        Array to be masked. 
        If 4D -> time series of 3D arrays. All images in the series will be equally masked
    mask : ndarray
        3D binary array
    Returns
    -------
    numpy masked array
    """
        
    # If the array to be masked has a 4th dimension (time) -> replicate the mask for all time points
    if array.ndim == 4: 
        mask = np.tile(mask[:,:,:,np.newaxis], [1,1,1,array.shape[3]])
        
    return np.ma.masked_where(mask==0, array)  


def _get_dcm_hdrs(dcm_dir: str, SN: int, N_echoes: int, EPI_bool: int = 1):

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    """
    Finds and reads the dicom headers associated w/ the first N_echoes echoes

    Returns
    -------
    list of dicom headers
        List of the dicom headers associated w/ the first N_echoes echoes
    """

    hdr_echoes_l = []

    # The main folder may be organized in different ways. Try one of them:
    try:
        subfolder_Path = os.path.join(dcm_dir, str(SN) + '/DICOM/')

        dcm_FNs = [FN for FN in os.listdir(
            subfolder_Path) if FN.endswith(".dcm")]

        for echo_idx in range(N_echoes):
            if EPI_bool:
                dcm_FN = [FN for FN in dcm_FNs if FN.split(
                    "-")[2] == str(echo_idx+1)][0]
            else:  # GRE FMap
                dcm_FN = dcm_FNs[echo_idx]

            hdr_echoes_l.append(dcmImage(subfolder_Path + dcm_FN).header)

    except:  # Try the other organization:

        DicomFolders = os.listdir(dcm_dir)
        # ex: 's7' -> 0007, 's11' -> 0011
        query = '0'*(4-len(str(SN))) + str(SN)
        DicomFolder = [
            fName for fName in DicomFolders if query in fName][0] + '/'

        DicomFNames = sorted(os.listdir(dcm_dir + DicomFolder))

        for echo_idx in range(N_echoes):

            DicomFName = DicomFNames[echo_idx]
            hdr_echoes_l.append(
                dcmImage(dcm_dir + DicomFolder + DicomFName).header)

    return hdr_echoes_l


def _get_TEs(dcm_hdr_echoes_l):
    """Reads the echo times from the dicom headers

    Parameters
    ----------
    dcm_hdr_echoes_l : list of dicom headers
       Dicom headers associated w/ the different echo times

    Returns
    -------
    ndarray
        Echo times (in s)
    """

    N_echoes = len(dcm_hdr_echoes_l)

    TEs = np.zeros(N_echoes)

    for echo_i in range(N_echoes):
        try: # Works for Prisma data
            TEs[echo_i] = dcm_hdr_echoes_l[echo_i].get(('0x18', '0x81'))*1e-3  # in s
        except: # Seems to work for Vida data
            TEs[echo_i] = dcm_hdr_echoes_l[echo_i].get(('5200','9230'))[0].get(('0018','9114'))[0].get(('0018','9082'))*1e-3  # in s

    return TEs


def _acq_params_PEdir(dcmHdr):
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    """
    Summarizes some acquisition parameters along the phase-encoding direction

    Returns
    -------
    t_esp : float
        EPI echo spacing (in s) = time between the acquisition of points w/ kx=0 from 2 consecutive lines
    t_acq_1_img : float
        Time (in s) spent in 1 traversal of k-space (ETL*t_esp)
    [ETL, BW_perPx_PE, total_BW_PE] : list
    """
            
    ETL = dcmHdr[('0x18','0x91')] # Echo Train Length (No. k-space lines)
    N_PE_steps = dcmHdr[('0018','0089')]
    assert ETL == N_PE_steps
    
    BW_perPx_PE = dcmHdr.get( ('0x19','0x1028') ) # Hz/pixel
    total_BW_PE = BW_perPx_PE * ETL # Hz
    t_esp = 1/total_BW_PE # s
    t_acq_1_img = t_esp*ETL # s
       
    return t_esp, t_acq_1_img, [ETL, BW_perPx_PE, total_BW_PE]


def _affine_per_slice(affine: npt.NDArray[np.float_], N_slcs: int, slc_thickness: float = 0) -> npt.NDArray[np.float_]:
    """
    Creates a 3D affine matrix: a 4x4 affine for each slice

    Parameters
    ----------
    affine : npt.NDArray[np.float_]
        4x4 voxel-to-world transformation (in mm)
    N_slcs : int
        Number of slices: 1 if single-slice acquisition; MB if SMS acquisition
    slc_thickness : float
        Slice thickness (in mm)

    Returns
    -------
    npt.NDArray[np.float_]
        3D array: 1 4x4 affine matrix for each slice
    """
    
    affine_perSlc = np.tile(affine[:,:,None], [1,1,N_slcs])
    
    if slc_thickness == 0:
        slc_ratio = 1
    else:
        slc_size = np.linalg.norm(affine[:3,2]) # Norm of the 3rd column
        slc_ratio = slc_thickness/slc_size

    for slc in range(N_slcs):

        slcCorner_voxCoords    = np.array([0,0,slc,1])
        slcCorner_worldCoords  = affine @ slcCorner_voxCoords
        affine_perSlc[:,3,slc] = slcCorner_worldCoords
        affine_perSlc[:,:,slc] = affine_perSlc[:,:,slc] @ np.diag([1,1,slc_ratio,1])

    return affine_perSlc


def interleave_MB_images(imgs_l, affines_l):

    rotation_mats     = []
    lowerLeft_corners = []
    slc_thicknesses   = []

    for img, affine in zip(imgs_l, affines_l):

        N_slcs = img.shape[2]

        rotation_mats.append(affine[:3,:3])
        slc_thicknesses.append(np.linalg.norm(affine[:3,2]))
        
        for slc in range(N_slcs):
            lowerLeft_corners.append( (affine @ np.array([0,0,slc,1]) )[:-1])  

    assert np.allclose(*rotation_mats)

    imgs = np.ma.concatenate(imgs_l, axis=2)
    lowerLeft_corners = np.stack(lowerLeft_corners, axis=0)
    Zs = lowerLeft_corners[:,2]

    set_indices = np.concatenate([[set_i]*img.shape[2] for set_i, img in enumerate(imgs_l)])
    
    sort_indides = np.argsort(Zs)

    imgs = imgs[:,:,sort_indides,...]
    lowerLeft_corners = lowerLeft_corners[sort_indides,:]
    set_indices = set_indices[sort_indides]
    Zs = Zs[sort_indides]

    new_slc_thicknesses = np.linalg.norm(np.diff(lowerLeft_corners, axis=0), axis=1)
    new_slc_thickness = np.mean(new_slc_thicknesses)

    # print(f"\nDistance between slices of individual niftis:          {slc_thicknesses}")
    # # print(f"Real distance between slices of interleaved niftis:    {new_slc_thicknesses}")
    # print(f"Imposed distance between slices of interleaved niftis: {new_slc_thickness:.2f}\n")

    new_affine = np.copy(affine)
    new_affine[:3,-1] = lowerLeft_corners[0]

    slc_ratio = new_slc_thickness / slc_thicknesses[-1]

    new_affine = new_affine @ np.diag([1,1,slc_ratio,1])

    return imgs, new_affine, Zs, set_indices


def summarize_acq_parameters(Mags_path: str, Phs_path: str, dcm_dir: str, N_echoes: int, EPI_bool=1, verbose=1):
    """
    Summarizes important information stored in the nii and dcm headers 

    Returns
    -------
    dict
        Dictionary with some acquisition parameters. Each key is a str (name of the parameter) 
        and each value is a tuple in the form (parameter_value, parameter_units)
    """

    AcqParams_dict = {}

    Mags, affine, niiHdr = load_nii(Mags_path)

    # Dimensions according to the nii headers:
    dims_nii = niiHdr["dim"][1:5]  # [Nrows, Ncols, Nslcs, Nimgs]

    pixdims_nii = niiHdr["pixdim"][1:5][0:-1] * \
        1e-3  # [res_row, res_col, res_slc] # in m

    # Scan numbers:
    Mag_SN = int(os.path.split(Mags_path)[1].split("_")[0][1:])
    
    if Phs_path != "":
        Ph_SN = int(os.path.split(Phs_path)[1].split("_")[0][1:])

    dcmHdrs_echoes_l = _get_dcm_hdrs(dcm_dir, Mag_SN, N_echoes, EPI_bool=EPI_bool)

    TEs = _get_TEs(dcmHdrs_echoes_l)

    AcqParams_dict["TEs"] = (TEs, "s")

    # Keep only the header associated w/ the first echo
    dcmHdr_e1 = dcmHdrs_echoes_l[0]

    t_obj = dcmHdr_e1[("0008", "0032")]
    AcqParams_dict["t_stamp"] = (
        t_obj.hour*3600 + t_obj.minute*60 + t_obj.second, "s")

    TR = dcmHdr_e1[('0018', '0080')] * 1e-3  # in s
    AcqParams_dict["TR"] = (TR, "s")

    if EPI_bool:
        t_esp, t_acq_1_img, [ETL, BW_perPx_PE, total_BW_PE] = _acq_params_PEdir(dcmHdr_e1)

    Nrows, Ncols = [int(re.sub("[^0-9]", "", n))
                    for n in dcmHdr_e1[('0051', '100b')].split("*")]

    # The axis of PE *w/ respect to the image*!
    PE_dir = dcmHdr_e1[('0018', '1312')]

    AcqParams_dict["N_row"] = (Nrows, "samples in the row dir")
    AcqParams_dict["N_col"] = (Ncols, "samples in the col dir")
    AcqParams_dict["PE_dir"] = (PE_dir, "Axis of PE w/ respect to the img")

    N_PE_steps = dcmHdr_e1[('0018', '0089')]

    if EPI_bool:
        assert ETL == N_PE_steps
        if ETL != AcqParams_dict["N_" + PE_dir.lower()][0]:
            if verbose:
                print("\nWarning: inconsistent information about the PE direction:\nPE_dir=" + PE_dir.lower() + " but N_" + PE_dir.lower() + "=" + str(AcqParams_dict["N_" + PE_dir.lower()][0]) + " != ETL=N_PE_steps=" + str(ETL) +
                      "\nPermuting Nrows=" + str(Nrows) + " with Ncols=" + str(Ncols))
            aux = Nrows
            Nrows = Ncols
            Ncols = aux

            AcqParams_dict["N_row"] = (Nrows, "samples in the row dir")
            AcqParams_dict["N_col"] = (Ncols, "samples in the col dir")

        AcqParams_dict["ETL"] = (ETL, "echoes / PE steps")

        # Check the consistency btw the info in the dcm and nii headers:
        assert np.all([AcqParams_dict["N_row"][0],
                      AcqParams_dict["N_col"][0]] == dims_nii[0:2])

    # https://dicom.innolitics.com/ciods/rt-dose/image-plane/00280030
    row_spacing, col_spacing = np.array(
        dcmHdr_e1[("0028", "0030")]) * 1e-3  # in m
    slc_thickness = dcmHdr_e1[('0018', '0050')]*1e-3  # in m
    AcqParams_dict["res_row"] = (row_spacing, "m")
    AcqParams_dict["res_col"] = (col_spacing, "m")

    # Check the consistency btw the info in the dcm and nii headers:
    pixdims_dcm = np.array(
        [row_spacing, col_spacing, slc_thickness]).astype(pixdims_nii.dtype)
    aux = np.round(pixdims_nii, 7) == np.round(pixdims_dcm, 7)
    if not np.all(aux):
        idx = np.argwhere(~aux)[:, 0]
        # The inconsistency is in the slc thickness
        assert len(idx) == 1 and idx[0] == 2
        if verbose:
            print("\nWarning: inconsistent information about slice thickness in dicom (%.1f mm) and nifti (%.1f mm) headers. Using that from the dicom header" % (
                pixdims_dcm[2]*1e3, pixdims_nii[2]*1e3))

    AcqParams_dict["slice_thickness"] = (slc_thickness, "m")

    FOV_rows_V1, FOV_cols_V1 = row_spacing * Nrows, col_spacing * Ncols

    FOV_rows_V2 = float(dcmHdr_e1[('0051', '100c')].split(
        'FoV ')[1].split("*")[0])*1e-3  # in m
    FOV_cols_V2 = float(dcmHdr_e1[('0051', '100c')].split(
        'FoV ')[1].split("*")[1])*1e-3  # in m

    if not [FOV_rows_V1, FOV_cols_V1] == [FOV_rows_V2, FOV_cols_V2]:
        if verbose:
            print("\nWarning: inconsistent information about FoVs in the dicom header. Using the computed (" +
                  str(FOV_rows_V1) + " x " + str(FOV_cols_V1) + "), rather than the read (" + str(FOV_rows_V2) + " x " + str(FOV_cols_V2) + ") ones")

    # Choose the computed FoVs:
    FOV_rows, FOV_cols = FOV_rows_V1, FOV_cols_V1

    AcqParams_dict["FOV_row"] = (FOV_rows, "m")
    AcqParams_dict["FOV_col"] = (FOV_cols, "m")

    BW_perPx_RO = dcmHdr_e1[(('0018', '0095'))]  # Hz/pixel
    t_acq_1_line = 1/BW_perPx_RO  # in s

    AcqParams_dict["BW_perPx_RO"] = (BW_perPx_RO, "Hz/pixel")
    AcqParams_dict["t_acq_1_line"] = (t_acq_1_line, "s")

    if EPI_bool:
        if ETL == Nrows:
            N_RO = Ncols
            N_PE = Nrows
            FOV_RO = FOV_cols
            FOV_PE = FOV_rows
        elif ETL == Ncols:
            N_RO = Nrows
            N_PE = Ncols
            FOV_RO = FOV_rows
            FOV_PE = FOV_cols
        else:
            print("\nCannot figure out N_PE and N_RO. Aborting")
            return

        AcqParams_dict["FOV_PE"] = (FOV_PE, "m")
        AcqParams_dict["N_PE"] = (N_PE, "samples in the PE dir")
        AcqParams_dict["FOV_RO"] = (FOV_RO, "m")
        AcqParams_dict["N_RO"] = (N_RO, "samples in the RO dir")

        total_BW_RO = BW_perPx_RO * N_RO
        # delta_nu = total_BW_RO/2
        t_dw = 1/total_BW_RO  # in s
        t_acq_1_line_v2 = t_dw * N_RO

        assert np.round(t_acq_1_line, 10) == np.round(t_acq_1_line_v2, 10)

        AcqParams_dict["BW_perPx_PE"] = (BW_perPx_PE, "Hz/pixel")
        AcqParams_dict["total_BW_PE"] = (total_BW_PE, "Hz")
        AcqParams_dict["t_esp"] = (t_esp, "s")
        AcqParams_dict["t_acq_1_img"] = (t_acq_1_img, "s")

        AcqParams_dict["total_BW_RO"] = (total_BW_RO, "Hz")
        AcqParams_dict["t_dw"] = (t_dw, "s")

    omega_L = 2*np.pi * dcmHdr_e1[('0018', '0084')] * 1e6  # in rad/s
    B0 = dcmHdr_e1[('0018', '0087')]  # in T
    gamma = omega_L / B0  # in rad/s/T

    if EPI_bool:
        G_PE = 2*np.pi / (gamma*FOV_PE*t_esp)  # in T/m
        AcqParams_dict["G_PE"] = (G_PE, "T/m")

        G_RO = (total_BW_RO * 2*np.pi) / (gamma*FOV_RO)  # in T/m
        AcqParams_dict["G_RO"] = (G_RO, "T/m")

    AcqParams_dict["omega_L"] = (omega_L, "rad/s")
    AcqParams_dict["B0"] = (B0, "T")
    AcqParams_dict["gamma_H"] = (gamma, "rad/s/T")

    flipAngle = dcmHdr_e1[('0018', '1314')]  # in ª
    AcqParams_dict["flipAngle"] = (flipAngle, "º")

    img_type = dcmHdr_e1[('0008', '0008')]
    Seq_variant = dcmHdr_e1[('0018', '0021')]  # Sequence variant
    # t_after_start = hdr.get(('0019','1016')) # ??
    AcqParams_dict["img_type"] = img_type
    AcqParams_dict["seq_variant"] = Seq_variant

    if EPI_bool:
        # https://github.com/CMRR-C2P/MB/issues/246
        img_comments = dcmHdr_e1[('0020', '4000')]
        shadow_hdr = dcmHdr_e1[('0029', '1010')]
        AcqParams_dict["img_comments"] = img_comments

        try:
            compressed_shadow_hdr = dict([(items[0], items[1]["value"])
                                         for items in shadow_hdr.items() if items[1]["value"] != None])
            AcqParams_dict["shadow_hdr"] = compressed_shadow_hdr
        except:
            pass


    # Create a 3D affine matrix: a 4x4 affine for each slice
    affine_perSlc = _affine_per_slice(affine, Mags.shape[2], slc_thickness*1e3)
    
    AcqParams_dict["affine"] = (affine_perSlc, "mm")

    return AcqParams_dict



def rescalePhase(PhaseImgs: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Rescales phase images to the interval [-pi,pi], if needed
    """

    minPhase, maxPhase = np.amin(PhaseImgs), np.amax(PhaseImgs)
    
    # Note on the range of phase images: https://github.com/CMRR-C2P/MB/issues/238
    
    if minPhase > - np.pi and maxPhase < np.pi:
        pass
    else:
        PhaseImgs = PhaseImgs - minPhase                # New range: [0, max-min]
        PhaseImgs = PhaseImgs / (maxPhase - minPhase)   # New range: [0, 1]
        PhaseImgs = PhaseImgs*2*np.pi - np.pi           # New range: [-pi, pi]

    return PhaseImgs


def calc_fft(t_series, Ts):

    N_fft = t_series.shape[0]
    
    Fs = 1/Ts
    f_axis = np.arange(-N_fft/2, N_fft/2)*(Fs/N_fft)

    FFT = fftshift(fft(t_series, axis=0), axes=0)

    return FFT, f_axis


def _get_electrodes_coords(elec_coords_path, cable_numbers):

    with open(elec_coords_path,"r") as file:
        lines = file.readlines()

    electrodes_coords = np.zeros([3,len(cable_numbers)]) # (3 x N_elec)

    for elec_i, elec_numb in enumerate(cable_numbers):
        for line in lines:
            if line.split("\n")[0].endswith(f"electrode {elec_numb}"):
                electrodes_coords[:,elec_i] = np.array(line.split()[0:3]).astype(float)
                break

    return electrodes_coords


def _electrodes_coords2img(ref_img, affine, electrodes_coords):

    X, Y, Z = np.meshgrid(np.arange(ref_img.shape[0]), np.arange(ref_img.shape[1]), np.arange(ref_img.shape[2]))

    Xs, Ys, Zs = X.flatten(order="C"), Y.flatten(order="C"), Z.flatten(order="C")

    gridPoints_vox = np.stack([Xs, Ys, Zs, np.ones(len(Xs))], axis=0) # (4 x N_points)
            
    # Convert the grid from voxel units to mm:
    gridPoints_mm = np.matmul(affine, gridPoints_vox)[0:3,:] # (3 x N_points)

    electrodes_img = np.zeros_like(ref_img)

    for elec_i in range(electrodes_coords.shape[1]):
        
        elec_coords = np.tile(electrodes_coords[:,elec_i,None], (1,gridPoints_mm.shape[1])) # (3 x N_points)
        assert elec_coords.shape == gridPoints_mm.shape

        # Find the voxel closest to the electrode center:
        voxel_elec_center = np.argmin(np.linalg.norm(elec_coords - gridPoints_mm, axis=0))
        gridPoint = (gridPoints_vox[0:3,voxel_elec_center]).astype(int)

        electrodes_img[tuple(gridPoint)] = 1

    kernel_size = 3
    electrodes_img = binary_dilation(electrodes_img, structure=np.ones([kernel_size,kernel_size,1])).astype(np.int8)

    return electrodes_img


def get_electrodes_img(elec_coords_path, cable_numbers, ref_img, affine):
    
    electrodes_coords = _get_electrodes_coords(elec_coords_path, cable_numbers)
    electrodes_img    = _electrodes_coords2img(ref_img, affine, electrodes_coords)

    return electrodes_img


def displayImg(Imgs, vmins, vmaxs, suptitle=None, titles=None,\
               figsize=None, grid=None, allTicks=0, maskTicks=0, xlabels="", ylabels="", xticks=None, yticks=None,\
               gridBool=0, gridColor="gray", APcolors="k", cmaps="gray", cbarBools=None, cbarTicks=None,\
               cbarLabels=None, cbarLocs="right", rot90=1, tLayout_pad=None, axis=1, showPlot=1, facecolor="white", axesOn=1):

    def _round_SigDigits(toRound, N_SigDigits):
    
        try: # If toRound is iterable
            rounded = [ round(num, N_SigDigits - int(np.floor(np.log10(abs(num)))) - 1) if num != 0 else 0 for num in toRound]
            if isinstance(toRound, np.ndarray): rounded = np.array(rounded)
        except:  # If toRound is a scalar
            rounded = round(toRound, N_SigDigits - int(np.floor(np.log10(abs(toRound)))) - 1) if toRound != 0 else 0
        
        return rounded 


    if not isinstance(Imgs, list): Imgs = [Imgs]
    
    N_imgs = len(Imgs)
            
    if not isinstance(vmaxs, (list, tuple, np.ndarray)): # a single value was provided
        vmaxs = np.ones(N_imgs)*vmaxs
    else:
        assert len(vmaxs) == N_imgs
    
    if not isinstance(vmins, list): # a single value was provided
        vmins = np.ones(N_imgs)*vmins
    else:
        assert len(vmins) == N_imgs
    
    if not isinstance(cmaps, list): cmaps = [cmaps]
    if len(cmaps) == 1: cmaps = [cmaps[0] for i in range(N_imgs)] # Same cmap for all subplots
    assert len(cmaps) == N_imgs

    if np.all(cbarBools != None):
        if not isinstance(cbarBools, list): cbarBools = [cbarBools]
        if len(cbarBools) == 1: cbarBools = [cbarBools[0] for i in range(N_imgs)] # Same bool for all subplots
        assert len(cbarBools) == N_imgs
    else: # If not specified, add cbar to all subplots
        cbarBools = [1 for i in range(N_imgs)]
        
    if np.all(cbarTicks != None):
        assert isinstance(cbarTicks, (list, tuple, np.ndarray))
        if len(cbarTicks)==0 or not isinstance(cbarTicks[0], (list, tuple, np.ndarray)): # If it's a simple list/array -> make it a list of lists/arrays
            cbarTicks = [cbarTicks for i in range(N_imgs)]
        assert len(cbarTicks) == N_imgs
        s=0  
    else:    
        cbarTicks = [_round_SigDigits([vmins[i],(vmins[i]+vmaxs[i])/2, vmaxs[i]], 2) for i in range(N_imgs)]
        s=0   
    if cbarLabels != None:
        if not isinstance(cbarLabels, list): cbarLabels = [cbarLabels]
        if len(cbarLabels) == 1: cbarLabels = [cbarLabels[0] for i in range(N_imgs)] # Same label for all colorbars
        assert len(cbarLabels) == N_imgs
    else:
        cbarLabels = ["" for i in range(N_imgs)]
    
    if isinstance(xlabels, str): xlabels = N_imgs * [xlabels]
    else:                        assert len(xlabels) == N_imgs
    
    if isinstance(ylabels, str): ylabels = N_imgs * [ylabels]
    else:                        assert len(ylabels) == N_imgs
    
    if np.all(xticks != None):
       if len(xticks) > 0 and not isinstance(xticks[0],(tuple,list,np.ndarray)): # A simple array-like was provided -> same ticks for all subplots
           xticks = [xticks for i in range(N_imgs)]
       else:
           assert len(xticks) == N_imgs and isinstance(xticks[0], (tuple,list,np.ndarray))

    if np.all(yticks != None):
       if len(yticks) > 0 and not isinstance(yticks[0],(tuple,list,np.ndarray)): # A simple array-like was provided -> same ticks for all subplots
           yticks = [yticks for i in range(N_imgs)]
       else:
           assert len(yticks) == N_imgs and isinstance(yticks[0], (tuple,list,np.ndarray))


    if isinstance(cbarLocs, str): cbarLocs = N_imgs * [cbarLocs]
    else:                         assert len(cbarLocs) == N_imgs
    
    if isinstance(APcolors, str): APcolors = N_imgs * [APcolors]
    else:                         assert len(APcolors) == N_imgs 

    if grid is None: grid = [1,N_imgs]
    l = 3.2
    figsize = (l*grid[1], l*grid[0]*0.95) if figsize is None else figsize

    fig, axes = plt.subplots(*grid, figsize=figsize)
    
    # grid = [1,1]
    if not isinstance(axes, np.ndarray): axes = np.array([[axes]]) # 2D
    
    # grid = [1,N_imgs]
    elif grid[0] == 1: axes = axes[np.newaxis,:] # Add the row dimension (only 1 row)
    
    # grid = [N_imgs,1]
    elif grid[1] == 1: axes = axes[:, np.newaxis] # Add the col dimension (only 1 col)
    
    # gridSpec = fig.add_gridspec(*grid)   
    # ax0 = fig.add_subplot(gridSpec[0,:]);   ax0.set_title("Cable Stray Fields", fontsize=9); ax0.axis('off')
    # ax1 = fig.add_subplot(gridSpec[1:3,:]); ax1.set_title("Residuals\nMB3, g=2", fontsize=9, pad=10); ax1.axis('off')
    
    if suptitle != None: 
        if isinstance(suptitle, str): plt.suptitle(suptitle)
        else:                         plt.suptitle(suptitle[0], fontsize=suptitle[1], y=suptitle[2] if len(suptitle)>=3 else 0.98, linespacing=suptitle[3] if len(suptitle)>=4 else 1.2)
    
    if titles != None: assert isinstance(titles, list) and len(titles) == N_imgs
    
    for row in range(grid[0]):
        for col in range(grid[1]):
            
            i = row * grid[1] + col # Image counter
            if i < N_imgs:
                #plt.subplot(*grid, i+1)
                ax = axes[row,col]
                if not axis: ax.set_axis_off()
                ax.set_facecolor(facecolor)
                
                fSize = 10
                if titles != None and not isinstance(titles[i],tuple): ax.set_title(titles[i])
                elif titles != None and isinstance(titles[i],tuple): 
                    fSize = titles[i][1] # Font size
                    ax.set_title(titles[i][0], fontsize=fSize, linespacing=titles[i][2] if len(titles[i])>=3 else 1.2,\
                                   pad=titles[i][3] if len(titles[i])>=4 else 6)
                    
                img = ax.imshow( np.rot90(Imgs[i], k=rot90, axes=(0,1)), vmin=vmins[i], vmax=vmaxs[i], cmap=cmaps[i])
                
                ax.set_xlabel(xlabels[i], fontsize=fSize-1 if fSize>7 else fSize); 
                ax.set_ylabel(ylabels[i], fontsize=fSize-1 if fSize>7 else fSize, rotation=90, labelpad=0)
                    
                if cbarBools[i]:
                    ax_divider = make_axes_locatable(ax)
                    cax = ax_divider.append_axes(cbarLocs[i], size="4%", pad="2%")
                    cbar = fig.colorbar(img, cax=cax, orientation="vertical" if (cbarLocs[i]=="right" or cbarLocs[i]=="left") else "horizontal")
                    
                    # cbar = fig.colorbar(img, ax=ax, use_gridspec=1)
                    cbar.set_ticks(cbarTicks[i])
                    # cbar.format('%.1e' if "e" in str(cbarTicks[i][0]) else '%.1f')
                    
                    cbar.ax.tick_params(labelsize=fSize-1 if fSize>7 else fSize)
                    cbar.set_label(cbarLabels[i][0] if isinstance(cbarLabels[i],tuple) else cbarLabels[i],\
                                    fontsize=fSize, labelpad=cbarLabels[i][1] if isinstance(cbarLabels[i],tuple) else 0.05)
                    
                # Add "A", "P" letters: 
                N_rows, N_cols = Imgs[i].shape[0:2]
                # if rot90: 
                #     ax.text(N_rows/2, 0.1*N_cols,  "A", color=APcolors[i], fontsize=fSize-1 if fSize!=None else None)
                #     ax.text(N_rows/2, 0.97*N_cols, "P", color=APcolors[i], fontsize=fSize-1 if fSize!=None else None) # Remember the 90º rotation
                # else: 
                #     ax.text(0.92*N_cols, N_rows/2, "A", color=APcolors[i], fontsize=fSize-1 if fSize!=None else None)
                #     ax.text(0.03*N_cols, N_rows/2, "P", color=APcolors[i], fontsize=fSize-1 if fSize!=None else None)
                
                if np.all(xticks != None): ax.set_xticks(xticks[i]); 
                if np.all(yticks != None): ax.set_yticks(yticks[i]);

                if not (np.all(xticks != None) or np.all(yticks != None)): 
                    if not allTicks:
                        ax.set_xticks([]); ax.set_yticks([])
                
                ax.tick_params(axis='both',labelsize=fSize-1)
                if gridBool: ax.grid(color=gridColor)
               
            else: # if i == N_img
                axes[row,col].remove()
               
            if not axesOn: axes[row,col].set_frame_on(False)
                
    try: # If the qt backend is being used:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
    except: pass
    

    if tLayout_pad != None:
        if not isinstance(tLayout_pad,list): tLayout_pad = [tLayout_pad,tLayout_pad]
        else: assert len(tLayout_pad) == 2 # Horizontal and Vertical padding
        plt.tight_layout(h_pad=tLayout_pad[0], w_pad=tLayout_pad[1])
    else:
         plt.tight_layout()
        
                
    if showPlot: plt.show()
    return fig


def tSeries_movie(img4D: npt.NDArray[np.float_], movie_path: str, dt_frames_ms: float, cmap: str = "gray", suptitle: str = ""):

    assert img4D.ndim == 4 # (row, col, slc, t)

    Nslcs = img4D.shape[2]
    l = 3 # "Size" of each suplot
    
    fig, axes = plt.subplots(1, Nslcs, figsize=(Nslcs*l, 1.1*l), tight_layout={"pad":0.5})
    fig.suptitle(suptitle)
    
    if not isinstance(axes, (tuple,list,np.ndarray)): axes = [axes]
    for ax in axes: ax.set_axis_off()
    
    q = 0.995
    vmax = max( [np.quantile(abs(img4D[:,:,slc,:]),q) for slc in range(Nslcs)] )
    vmin = 0 if cmap == "gray" else -vmax
    d = 0.075 # For placing the text box
    
    frames_l = []
    for t_i in range(img4D.shape[3]):
        
        frame = []
        text = axes[0].text(d,1-d, str(t_i), animated=True, ha='left', va='top', color="k", transform=axes[0].transAxes, backgroundcolor="white")
        frame += [text]
        
        for slc in range(Nslcs):
            
            img = axes[slc].imshow(np.rot90(img4D[:,:,slc,t_i]), cmap="gray", vmin=vmin, vmax=vmax, animated=True)
            frame += [img]
            
        frames_l.append(frame)

    ani = animation.ArtistAnimation(fig, frames_l, interval=dt_frames_ms, blit=True, repeat_delay=1000)
    ani.save(movie_path)
    
    plt.close()


def read_used_coils(dcm_dir, nii_FNs):

    from dicom_parser import Image as dcmImage

    SNs = np.array([int(fn.split("_")[0][1:]) for fn in nii_FNs])
    nii_FNs = nii_FNs[np.argsort(SNs)]

    output_str = ""

    for nii_FN, SN in zip(nii_FNs,SNs):

        FNs_in_dcm_dir = os.listdir(f"{dcm_dir}{SN}/DICOM/")
        FN = [fn for fn in FNs_in_dcm_dir if f"-{SN}-1-" in fn]

        assert len(FN) == 1, FN; FN = FN[0]

        hdr = dcmImage(f"{dcm_dir}{SN}/DICOM/{FN}").header
        used_coils = hdr[('0051', '100f')]

        print(nii_FN)
        print("Rx coils:", used_coils, "\n")

        output_str += f"{nii_FN}\nRx coils: {used_coils}\n\n"

    return output_str