#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CreateSimulatedDBzc.py
Converted from CreateSimulatedDBzc.ipynb

Description:
This script simulates tDCS fields using SimNIBS, calculates the Magnetic Flux Density (Bz)
using the Biot-Savart law, and saves the result as a NIfTI file simulating a measurement
across 5 slices.
"""

#%% Imports
import os
import sys
import copy
import numpy as np
import nibabel as nib
import scipy.io
import matplotlib.pyplot as plt

try:
    from simnibs import sim_struct
    from simnibs.simulation import biot_savart as BS
    from simnibs.simulation import fem
    from simnibs.utils import transformations
except ImportError as e:
    print("Hata: SimNIBS kütüphanesi bulunamadı. Lütfen SimNIBS ortamını aktif ettiğinizden emin olun.")
    print(f"Detay: {e}")
    sys.exit(1)

def main():
    print("--- Simülasyon Başlatılıyor ---")

    #%% Configuration & Paths
    # UYARI: Bu dosya yollarını kendi sisteminize göre düzenleyin!
    base_project_path = "/Users/nalansonverdi/Desktop/unt"
    vers_path = "/Users/nalansonverdi/Desktop/unt/outputs"
    m2m_path = "/Users/nalansonverdi/simnibs-4.5.0/examples/m2m_ernie"
    
    session_mat_path = "/Users/nalansonverdi/Desktop/unt/simnibs_simulation_20260130-152929.mat"
    
    print(f"Session mat dosyası yükleniyor: {session_mat_path}")
    
    # Load session mat
    session_mat = scipy.io.loadmat(session_mat_path, struct_as_record=True, squeeze_me=False)

    # Update paths in session structure (environment specific adjustments)
    session_mat["subpath"] = np.array([m2m_path])
    session_mat["eeg_cap"] = np.array("/Users/nalansonverdi/simnibs-4.5.0/examples/m2m_ernie/eeg_positions/EEG10-10_UI_Jurak_2007.csv")
    session_mat["fnamehead"] = np.array("/Users/nalansonverdi/simnibs-4.5.0/examples/m2m_ernie/ernie_custom.msh")
    session_mat["pathfem"] = np.array([vers_path])
    session_mat["fname_tensor"] = np.array("/Users/nalansonverdi/simnibs-4.5.0/examples/m2m_ernie/DTI_coregT1_tensor.nii.gz")

    # Initialize Session
    session = sim_struct.SESSION(session_mat)
    session._prepare()
    tdcs = session.poslists[0]
    tdcs.postprocess = 'vEJ'

    #%% Conductivity Settings
    print("İletkenlik değerleri ayarlanıyor...")
    
    # Print existing conductivities (optional debugging)
    # for i, cond in enumerate(tdcs.cond):
    #     if cond.value != None: print(i, cond.name, cond.value)

    # !!! IMPORTANT CHANGES !!!
    # Specific tissue indices depend on the mesh structure (ernie_custom)
    try:
        tdcs.cond[2].value = 0.80 # cCSF 
        tdcs.cond[2].name = "cCSF"
        tdcs.cond[6].value = 0.01 # CB
        tdcs.cond[7].value = 0.01 # SB
        
        # Check if index 12 exists (Ventricular CSF)
        if len(tdcs.cond) > 12:
            tdcs.cond[12].value = 1.79 # Added ventricular CSF (Gregersen et al., 2024)
            tdcs.cond[12].name = "vCSF"
    except IndexError as e:
        print(f"Hata: İletkenlik indeksleri mesh yapısıyla uyuşmuyor: {e}")

    #%% Load Masks
    print("Maskeler yükleniyor...")
    # Loading and concatenating masks for 5 slices
    mask_files = [
        "/Users/nalansonverdi/Desktop/unt/masks/T1_brain_mask_2mm_ROI0.nii.gz",
        "/Users/nalansonverdi/Desktop/unt/masks/T1_brain_mask_2mm_ROI1.nii.gz",
        "/Users/nalansonverdi/Desktop/unt/masks/T1_brain_mask_2mm_ROI2.nii.gz",
        "/Users/nalansonverdi/Desktop/unt/masks/T1_brain_mask_2mm_ROI3.nii.gz",
        "/Users/nalansonverdi/Desktop/unt/masks/T1_brain_mask_2mm_ROI4.nii.gz"
    ]
    
    masks_nib_l = []
    for f in mask_files:
        full_path = os.path.join(base_project_path, f)
        if os.path.exists(full_path):
            masks_nib_l.append(nib.load(full_path))
        else:
            print(f"Uyarı: Maske dosyası bulunamadı: {full_path}")
            sys.exit(1)

    # Reading n_voxels and affine from masks (Assuming n_voxels is the same for all slices)
    n_voxels = masks_nib_l[0].get_fdata().shape[:3]
    if len(n_voxels) == 2:
        n_voxels = n_voxels + (1,)
    
    affines_l = [mask_nib.affine for mask_nib in masks_nib_l]

    #%% FEM Simulation
    print("FEM Matrisi oluşturuluyor ve çözülüyor (J hesaplanıyor)...")
    
    tdcslist = copy.deepcopy(tdcs)
    mesh, electrode_surfaces = tdcslist._place_electrodes()
    cond = tdcslist.cond2elmdata(mesh=mesh, logger_level=10)
    
    # Calculate Potentials (v)
    v = fem.tdcs(mesh, cond, tdcslist.currents, np.unique(electrode_surfaces))
    
    # Calculate Fields (J)
    m = fem.calc_fields(v, 'J', cond=cond)
    J_ed = m.field['J']

    #%% Biot-Savart Calculation (B Field)
    print("Biot-Savart yasası ile B alanı hesaplanıyor...")
    
    mesh = J_ed.mesh
    calc_res = 2 
    
    # Computational Domain
    domain = BS._comp_domain(mesh, n_voxels, affines_l[0])
    
    # Voxelize J
    J_vol, affine_vol = BS._voxelize(J_ed, domain, calc_res) 
    
    # Calculate B in the whole domain
    B_vol = BS._bs_ft(J_vol, calc_res)

    #%% Extract Slices and Save NIfTI
    print("Dilimler çıkarılıyor ve kaydediliyor...")
    
    # Create a modified affine for the output file
    output_affine = copy.copy(affines_l[0])
    output_affine[2, 1] = 10 # just to have 5 slices in one file
    
    # Initialize output array (dims: X, Slices, Y based on notebook logic)
    # Notebook dimensions: (128, 5, 104)
    Bz = np.zeros((128, 5, 104))
    
    for i in range(5):
        print(f"Slice {i} işleniyor...")
        B = transformations.volumetric_affine(
                (B_vol, affine_vol), np.eye(4), affines_l[i], n_voxels,
                intorder=1, keep_vector_length=False
            )
        
        # Extract Y component of the B field (index 1) based on notebook code: Bz_tmp=B[:,:,:,1]
        # Note: Variable name is Bz but extracting index 1 (usually By in RAS? Depends on SimNIBS frame).
        # Keeping consistent with notebook logic.
        Bz_tmp = B[:, :, :, 1]
        
        mask = masks_nib_l[i].get_fdata()
        # Mask out voxels outside the brain
        Bz_tmp[mask == 0] = 0 
        
        Bz[:, i, :] = np.squeeze(Bz_tmp)

    output_filename = "Dbzc_Meas_5Slices.nii.gz"
    output_filename = os.path.join(base_project_path, 'Dbzc_Meas_5Slices.nii.gz')
    print(f"Dosya kaydediliyor: {output_filename}")
    nib.save(nib.Nifti1Image(Bz, output_affine), output_filename)

    #%% Plotting
    print("Sonuçlar çiziliyor ve kaydediliyor...")
    fig = plt.figure(figsize=(15, 3)) # Figure nesnesini değişkene atayalım
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(Bz[:, i, :], vmin=-0.5e-9, vmax=0.5e-9, cmap="bwr")
        plt.title(f"Slice {i}")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Kaydetme işlemini SHOW'dan ÖNCE yapmalısın
    plot_filename = os.path.join(base_project_path, 'Bz_Slices_Output.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Görsel kaydedildi: {plot_filename}")

    # Ekranda görmek istiyorsan en son bunu çağır
    plt.show() 
    
    print("İşlem tamamlandı.")

if __name__ == "__main__":
    main()