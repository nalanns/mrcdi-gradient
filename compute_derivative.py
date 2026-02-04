import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

# --- Lütfen gradient_utils dosyanızın bu script ile aynı klasörde olduğundan emin olun ---
try:
    from gradient_utils import xGradient, yGradient
except ImportError:
    print("UYARI: gradient_utils bulunamadı. Lütfen dosya yolunu kontrol edin.")
    # Kodun çökmemesi için geçici tanımlar (Sizde dosya olduğu için burası çalışmayacaktır)
    def xGradient(img, res): return np.gradient(img, res, axis=0)
    def yGradient(img, res): return np.gradient(img, res, axis=1)

# 1. Veriyi Yükle ve Hesapla
nifti_path = "/Users/nalansonverdi/Desktop/unt/Dbzc_Meas_5Slices.nii.gz"
nifti_img = nib.load(nifti_path)
data = nifti_img.get_fdata() 
res_x = 0.002

gradients = np.zeros_like(data)
y_gradients = np.zeros_like(data)

# İlk türevleri hesapla (Gradient)
for i in range(data.shape[1]):
    gradients[:, i, :] = xGradient(data[:, i, :], res_x)
    y_gradients[:, i, :] = yGradient(data[:, i, :], res_x)

# Maske oluşturma ve kenarları aşındırma
mask = np.abs(data) > 0
eroded_mask = np.zeros_like(mask)
for i in range(data.shape[1]):
    eroded_mask[:, i, :] = binary_erosion(mask[:, i, :], iterations=2)

gradients[~eroded_mask] = 0
y_gradients[~eroded_mask] = 0

# Norm Gradient Hesapla
norm_gradients = np.sqrt(gradients**2 + y_gradients**2)
norm_gradients[~eroded_mask] = 0

# ---------------------------------------------------------
# DÜZELTİLEN KISIM: Laplacian Hesapla ve Ölçek Belirle
# ---------------------------------------------------------
laplacian = np.zeros_like(data)
for i in range(data.shape[1]):
    # Laplacian: Gradientlerin tekrar türevi (d^2/dx^2 + d^2/dy^2)
    grad_xx = xGradient(gradients[:, i, :], res_x)
    grad_yy = yGradient(y_gradients[:, i, :], res_x)
    laplacian[:, i, :] = grad_xx + grad_yy

laplacian[~eroded_mask] = 0

# Laplacian için Dinamik Vmax Hesaplama
# Sabit 1e-8 yerine, verinin %98'ini kapsayan değeri buluyoruz.
# Sadece maske içindeki değerleri dikkate alıyoruz.
valid_laplacian_vals = np.abs(laplacian[eroded_mask])
if valid_laplacian_vals.size > 0:
    vmax_lap_dynamic = np.percentile(valid_laplacian_vals, 98)
else:
    vmax_lap_dynamic = 1e-8 # Veri boşsa fallback

print(f"Laplacian Max Değeri: {np.max(laplacian)}")
print(f"Laplacian Görselleştirme için hesaplanan Vmax: {vmax_lap_dynamic}")

# ---------------------------------------------------------

# Görselleştirme Verilerini Hazırla
plot_data_x = np.abs(gradients)
plot_data_y = np.abs(y_gradients)
plot_data_norm = norm_gradients
plot_data_lap = np.abs(laplacian)

plt.style.use('default') 

# 2. GÖRSELLEŞTİRME - X Gradient (Aynen Korundu)
fig, axes = plt.subplots(1, 5, figsize=(22, 6), facecolor='white')
vmin = 0
vmax = 1e-8

for i in range(5):
    slice_data = plot_data_x[:, i, :].T
    rotated_data = np.rot90(slice_data, k=-3)
    im = axes[i].imshow(rotated_data, cmap='viridis', 
                        origin='lower', vmin=vmin, vmax=vmax)
    axes[i].set_title(f"X Gradient Slice {i}", fontsize=13, fontweight='bold', pad=15)
    axes[i].axis('off')

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Gradient Magnitude (1/m)', rotation=90, labelpad=20, fontsize=12)
plt.suptitle("Dbzc X-Gradients Analysis", fontsize=18, y=0.98)
plt.savefig("/Users/nalansonverdi/Desktop/unt/Dbzc_Gradients_X_WhiteBG.png", dpi=300, bbox_inches='tight', facecolor='white')

# 3. GÖRSELLEŞTİRME - Y Gradient (Aynen Korundu)
fig_y, axes_y = plt.subplots(1, 5, figsize=(22, 6), facecolor='white')
vmax_y = 1e-8

for i in range(5):
    slice_data = plot_data_y[:, i, :].T
    rotated_data = np.rot90(slice_data, k=-3)
    im_y = axes_y[i].imshow(rotated_data, cmap='viridis', 
                        origin='lower', vmin=vmin, vmax=vmax_y)
    axes_y[i].set_title(f"Y Gradient Slice {i}", fontsize=13, fontweight='bold', pad=15)
    axes_y[i].axis('off')

fig_y.subplots_adjust(right=0.88)
cbar_ax_y = fig_y.add_axes([0.91, 0.15, 0.015, 0.7])
cbar_y = fig_y.colorbar(im_y, cax=cbar_ax_y)
cbar_y.set_label('Gradient Magnitude (1/m)', rotation=90, labelpad=20, fontsize=12)
plt.suptitle("Dbzc Y-Gradients Analysis", fontsize=18, y=0.98)
plt.savefig("/Users/nalansonverdi/Desktop/unt/Dbzc_Gradients_Y_WhiteBG.png", dpi=300, bbox_inches='tight', facecolor='white')

# 4. GÖRSELLEŞTİRME - Norm Gradient (Aynen Korundu)
fig_norm, axes_norm = plt.subplots(1, 5, figsize=(22, 6), facecolor='white')
vmax_norm = 1e-8

for i in range(5):
    slice_data = plot_data_norm[:, i, :].T
    rotated_data = np.rot90(slice_data, k=-3)
    im_norm = axes_norm[i].imshow(rotated_data, cmap='viridis', 
                        origin='lower', vmin=vmin, vmax=vmax_norm)
    axes_norm[i].set_title(f"Norm Gradient Slice {i}", fontsize=13, fontweight='bold', pad=15)
    axes_norm[i].axis('off')

fig_norm.subplots_adjust(right=0.88)
cbar_ax_norm = fig_norm.add_axes([0.91, 0.15, 0.015, 0.7])
cbar_norm = fig_norm.colorbar(im_norm, cax=cbar_ax_norm)
cbar_norm.set_label('Gradient Magnitude (1/m)', rotation=90, labelpad=20, fontsize=12)
plt.suptitle("Dbzc Norm-Gradients Analysis", fontsize=18, y=0.98)
plt.savefig("/Users/nalansonverdi/Desktop/unt/Dbzc_Gradients_Norm_WhiteBG.png", dpi=300, bbox_inches='tight', facecolor='white')

# 5. GÖRSELLEŞTİRME - Laplacian (GÜNCELLENDİ)
fig_lap, axes_lap = plt.subplots(1, 5, figsize=(22, 6), facecolor='white')

# Burada hesapladığımız dinamik vmax değerini kullanıyoruz
vmax_lap = vmax_lap_dynamic

for i in range(5):
    slice_data = plot_data_lap[:, i, :].T
    rotated_data = np.rot90(slice_data, k=-3)
    # vmin=0, vmax=hesaplanan_deger
    im_lap = axes_lap[i].imshow(rotated_data, cmap='viridis', 
                        origin='lower', vmin=0, vmax=vmax_lap)
    
    axes_lap[i].set_title(f"Laplacian Slice {i}", fontsize=13, fontweight='bold', pad=15)
    axes_lap[i].axis('off')

fig_lap.subplots_adjust(right=0.88)
cbar_ax_lap = fig_lap.add_axes([0.91, 0.15, 0.015, 0.7])
cbar_lap = fig_lap.colorbar(im_lap, cax=cbar_ax_lap)
cbar_lap.set_label('Laplacian Magnitude (1/m^2)', rotation=90, labelpad=20, fontsize=12)

plt.suptitle("Dbzc Laplacian Analysis", fontsize=18, y=0.98)
plt.savefig("/Users/nalansonverdi/Desktop/unt/Dbzc_Laplacian_WhiteBG.png", dpi=300, bbox_inches='tight', facecolor='white')

plt.show()