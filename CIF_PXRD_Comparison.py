#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

try:
    from pymatgen.io.cif import CifParser
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
except ImportError:
    print("Error: 'pymatgen' library not found. Please install it using 'pip install pymatgen'")
    exit(1)

# --- GLOBAL CONFIGURATION ---
# It's possible to change Theta range, step size, and X-ray wavelength here for more flexibility
THETA_MIN = 5.0
THETA_MAX = 90.0
STEP = 0.01
WAVELENGTH = 1.54056  # Cu K-alpha radiation
X_GRID = np.arange(THETA_MIN, THETA_MAX, STEP)

# Shift parameters to account for unit cell volume tolerance
MAX_SHIFT = 1.5       
SHIFT_STEP = 0.05     

def generate_clean_pattern(cif_path, calc):
    """Computes the theoretical XRD pattern and applies a Pseudo-Voigt broadening."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser = CifParser(cif_path, occupancy_tolerance=100.0)
            structure = parser.parse_structures(primitive=False)[0]
            pattern = calc.get_pattern(structure, two_theta_range=(THETA_MIN, THETA_MAX))
    except Exception as e:
        print(f"  [!] Error processing {os.path.basename(cif_path)}: {e}")
        return None

    intensity = np.zeros_like(X_GRID)
    
    # Caglioti empirical parameters for FWHM
    # You can change these parameters with lower value to simulate experimental patterns with sharper peaks, or higher values for more broadening
    u, v, w = 0.04, -0.004, 0.04

    for two_theta, peak_int in zip(pattern.x, pattern.y):
        if two_theta < THETA_MIN or two_theta > THETA_MAX:
            continue
            
        rad_theta = np.radians(two_theta / 2)
        fwhm_sq = u * np.tan(rad_theta)**2 + v * np.tan(rad_theta) + w
        fwhm = max(np.sqrt(abs(fwhm_sq)), 0.01)
        
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        gamma = fwhm / 2
        
        idx_min = int(max(0, (two_theta - 2.5 - THETA_MIN) / STEP))
        idx_max = int(min(len(X_GRID), (two_theta + 2.5 - THETA_MIN) / STEP))
        if idx_min >= idx_max: 
            continue
            
        dx = X_GRID[idx_min:idx_max] - two_theta
        gauss = np.exp(-0.5 * (dx / sigma)**2)
        lorentz = 1 / (1 + (dx / gamma)**2)
        profile = 0.5 * gauss + 0.5 * lorentz
        
        intensity[idx_min:idx_max] += peak_int * profile

    if np.max(intensity) > 0:
        intensity = (intensity / np.max(intensity)) * 100.0
        
    return intensity

def calc_cross_correlation(y_ref, y_cand):
    """Calculates the de Gelder Similarity and Dissimilarity."""
    norm_ref = np.sqrt(np.sum(y_ref**2))
    norm_cand = np.sqrt(np.sum(y_cand**2))
    
    if norm_ref == 0 or norm_cand == 0:
        return 0.0, 1.0 
        
    dot_product = np.sum(y_ref * y_cand)
    similarity = dot_product / (norm_ref * norm_cand)
    similarity = min(max(similarity, 0.0), 1.0) 
    
    dissimilarity = 1.0 - similarity
    return similarity, dissimilarity

def find_best_shift(y_ref, y_cand):
    """Finds the 2-theta shift that minimizes dissimilarity."""
    best_diss = 1.0
    best_sim = 0.0
    best_shift = 0.0
    best_y_aligned = y_cand

    shifts = np.arange(-MAX_SHIFT, MAX_SHIFT + SHIFT_STEP, SHIFT_STEP)
    
    for shift in shifts:
        shifted_x = X_GRID + shift
        interp_func = interp1d(shifted_x, y_cand, kind='linear', bounds_error=False, fill_value=0)
        y_cand_shifted = interp_func(X_GRID)
        
        sim, diss = calc_cross_correlation(y_ref, y_cand_shifted)
        
        if diss < best_diss:
            best_diss = diss
            best_sim = sim
            best_shift = shift
            best_y_aligned = y_cand_shifted
            
    return best_sim, best_diss, best_shift, best_y_aligned

def save_xy(x, y, filepath):
    np.savetxt(filepath, np.column_stack((x, y)), fmt='%.4f\t%.6f', header='2Theta\tIntensity', comments='')

def plot_comparison(y_ref, y_cand_orig, y_cand_shifted, best_shift, diss_score, ref_name, cand_name, output_png):
    plt.figure(figsize=(10, 6))
    plt.plot(X_GRID, y_ref, color='black', linewidth=1.5, label=f'Ref: {ref_name}')
    plt.plot(X_GRID, y_cand_orig + 120, color='red', alpha=0.5, linestyle='--', label='Original Candidate')
    plt.plot(X_GRID, y_cand_shifted + 120, color='blue', linewidth=1.5, 
             label=f'Optimized (Shift: {best_shift:+.2f}°, Diss: {diss_score:.4f})')
    
    plt.title(f'XRD Comparison: {ref_name} vs {cand_name}', fontsize=14, fontweight='bold')
    plt.xlabel(r'$2\theta$ Angle (Degrees)', fontsize=12)
    plt.ylabel('Relative Intensity', fontsize=12)
    plt.xlim(THETA_MIN, THETA_MAX)
    plt.yticks([]) 
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()

def process_single_candidate(cand_cif, y_ref, ref_name, calc, output_folder):
    """Handles the processing, plotting, and saving for a single candidate."""
    cand_name = os.path.basename(cand_cif)
    print(f"  -> Comparing with {cand_name}...")
    
    y_cand_orig = generate_clean_pattern(cand_cif, calc)
    if y_cand_orig is None:
        return None

    save_xy(X_GRID, y_cand_orig, os.path.join(output_folder, f"CAND_Orig_{cand_name}.xy"))

    best_sim, best_diss, best_shift, y_cand_shifted = find_best_shift(y_ref, y_cand_orig)

    save_xy(X_GRID, y_cand_shifted, os.path.join(output_folder, f"CAND_Shifted_{cand_name}.xy"))
    
    report_filename = os.path.join(output_folder, f"Plot_{cand_name}.png")
    plot_comparison(y_ref, y_cand_orig, y_cand_shifted, best_shift, best_diss, ref_name, cand_name, report_filename)
    
    verdict = "SAME POLYMORPH" if best_diss < 0.03 else "DIFFERENT POLYMORPHS"
    
    return {
        'name': cand_name,
        'similarity': best_sim,
        'dissimilarity': best_diss,
        'shift': best_shift,
        'verdict': verdict
    }

def print_result_block(r):
    """Helper function to print a single result block consistently."""
    threshold_info = "(Score < 0.03)" if r['dissimilarity'] < 0.03 else "(Score >= 0.03)"
    print("=============================================")
    print(f" COMPARISON METRICS - {r['name']}")
    print("=============================================")
    print(f" Raw Similarity (0 to 1)       : {r['similarity']:.4f}")
    print(f" Dissimilarity Score           : {r['dissimilarity']:.4f}")
    print(f" Applied Angular Shift         : {r['shift']:+.2f}°")
    print("---------------------------------------------")
    print(f" VERDICT: {r['verdict']} {threshold_info}")
    print("=============================================")

def main():
    # ==========================================
    # --- CONFIGURATION ZONE ---
    # ==========================================
    REFERENCE_CIF = "Quercetin/NAFZEC02.cif"
    
    # Choose mode: "single" or "batch"
    ANALYSIS_MODE = "batch" 
    
    # If "single", specify the candidate file:
    CANDIDATE_FILE = "Quercetin/Struttura_Deep_Relax.cif"
    
    # If "batch", specify the folder containing candidate CIFs:
    CANDIDATE_FOLDER = "Quercetin/Top50_Best_Refined" 

    # Name of OUTPUT_FOLDER will be auto-generated based on mode and candidate names
    if ANALYSIS_MODE == "single":
        OUTPUT_FOLDER = f"Comparison_Results_{os.path.splitext(os.path.basename(CANDIDATE_FILE))[0]}" 
    else:
        folder_name = os.path.basename(os.path.normpath(CANDIDATE_FOLDER))
        OUTPUT_FOLDER = f"Comparison_Results_Batch_{folder_name}"
    # ==========================================

    if not os.path.exists(REFERENCE_CIF):
        print(f"[!] Error: Reference file '{REFERENCE_CIF}' not found.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    calc = XRDCalculator(wavelength=WAVELENGTH)
    ref_name = os.path.basename(REFERENCE_CIF)

    print(f"-> Processing Reference: {ref_name}...")
    y_ref = generate_clean_pattern(REFERENCE_CIF, calc)
    if y_ref is None:
        print("[!] Failed to process reference pattern. Exiting.")
        return
        
    save_xy(X_GRID, y_ref, os.path.join(OUTPUT_FOLDER, f"REF_{ref_name}.xy"))

    results = []

    if ANALYSIS_MODE == "single":
        if not os.path.exists(CANDIDATE_FILE):
            print(f"[!] Error: Candidate file '{CANDIDATE_FILE}' not found.")
            return
        res = process_single_candidate(CANDIDATE_FILE, y_ref, ref_name, calc, OUTPUT_FOLDER)
        if res: results.append(res)

    elif ANALYSIS_MODE == "batch":
        if not os.path.exists(CANDIDATE_FOLDER):
            print(f"[!] Error: Folder '{CANDIDATE_FOLDER}' not found.")
            return
            
        cif_files = glob.glob(os.path.join(CANDIDATE_FOLDER, "*.cif"))
        # Exclude the reference if it happens to be in the same folder
        cif_files = [f for f in cif_files if os.path.abspath(f) != os.path.abspath(REFERENCE_CIF)]
        
        print(f"\n-> Found {len(cif_files)} candidates in '{CANDIDATE_FOLDER}'. Starting batch process...")
        for cif in cif_files:
            res = process_single_candidate(cif, y_ref, ref_name, calc, OUTPUT_FOLDER)
            if res: results.append(res)
    else:
        print("[!] Invalid ANALYSIS_MODE. Choose 'single' or 'batch'.")
        return

    # --- Generate TXT Report & Terminal Output ---
    if results:
        report_path = os.path.join(OUTPUT_FOLDER, "Summary_Report.txt")
        
        # Sort results by dissimilarity (best matches first)
        results.sort(key=lambda x: x['dissimilarity'])
        
        # 1. Write everything to the text file
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("XRD COMPARISON REPORT\n")
            f.write(f"Reference Structure: {ref_name}\n\n")
            
            for r in results:
                threshold_info = "(Score < 0.03)" if r['dissimilarity'] < 0.03 else "(Score >= 0.03)"
                
                f.write("=============================================\n")
                f.write(f" COMPARISON METRICS - {r['name']}\n")
                f.write("=============================================\n")
                f.write(f" Raw Similarity (0 to 1)       : {r['similarity']:.4f}\n")
                f.write(f" Dissimilarity Score           : {r['dissimilarity']:.4f}\n")
                f.write(f" Applied Angular Shift         : {r['shift']:+.2f}°\n")
                f.write("---------------------------------------------\n")
                f.write(f" VERDICT: {r['verdict']} {threshold_info}\n")
                f.write("=============================================\n\n")

        # 2. Terminal Printing Logic
        print("\n" + "#"*50)
        print(" ANALYSIS COMPLETE - TERMINAL SUMMARY")
        print("#"*50 + "\n")

        total_candidates = len(results)

        if ANALYSIS_MODE == "single" or total_candidates <= 6:
            # Print all if 6 or fewer
            for r in results:
                print_result_block(r)
                print()
        else:
            # Print Top 5
            print(">>> TOP 5 BEST MATCHES <<<\n")
            for r in results[:5]:
                print_result_block(r)
                print()
            
            # Print placeholder for omitted files
            omitted = total_candidates - 6
            print(f" ... [ {omitted} candidates omitted for brevity. Full list in Summary_Report.txt ] ... \n")
            
            # Print the worst match
            print(">>> MOST DIFFERENT CANDIDATE <<<\n")
            print_result_block(results[-1])
            print()

        print(f"[*] All {total_candidates} processed files and the complete Summary Report are saved in: ./{OUTPUT_FOLDER}/\n")

if __name__ == "__main__":
    main()
