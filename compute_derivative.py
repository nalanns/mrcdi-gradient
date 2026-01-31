import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from gradient_utils import xGradient  # Import the derivative function

# Load your NIfTI file (output from CreateSimulatedDBzc.py)
nifti_path = "/Users/nalansonverdi/Desktop/unt/Dbzc_Meas_5Slices.nii.gz"
nifti_img = nib.load(nifti_path)
data = nifti_img.get_fdata()  # Shape: (128, 5, 104) - x, slices, y
affine = nifti_img.affine

# Voxel resolution (from your script: 2mm = 0.002m; adjust if different)
res_x = 0.002  # meters

# Compute gradient for each slice (loop over the 5 slices)
gradients = np.zeros_like(data)  # To store results: shape (128, 5, 104)
for slice_idx in range(data.shape[1]):  # 5 slices
    slice_2d = data[:, slice_idx, :]  # Extract 2D slice: (128, 104)
    # Optional: Mask if you have a brain mask (e.g., from your masks/)
    # mask = nib.load("path/to/mask.nii.gz").get_fdata()[:, slice_idx, :]
    # slice_2d = np.ma.masked_where(mask == 0, slice_2d)
    
    grad_2d = xGradient(slice_2d, res_x)  # Compute gradient (output in 1/m)
    gradients[:, slice_idx, :] = grad_2d

# Save the gradient as a new NIfTI (optional, for inspection)
grad_nifti = nib.Nifti1Image(gradients, affine)
nib.save(grad_nifti, "/Users/nalansonverdi/Desktop/unt/Dbzc_Gradients.nii.gz")

# Plot like in your original script (adjust vmin/vmax based on gradient range)
fig = plt.figure(figsize=(15, 3))
vmin, vmax = np.min(gradients), np.max(gradients)
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(gradients[:, i, :], vmin=vmin, vmax=vmax, cmap="bwr")  # Use actual range
    plt.title(f"Gradient Slice {i}")
    plt.axis('off')
plt.tight_layout()
plt.savefig("/Users/nalansonverdi/Desktop/unt/Dbzc_Gradients_Plot.png", dpi=300, bbox_inches='tight')
plt.show()