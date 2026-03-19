# XRD Pattern Cross-Correlation Matcher

This Python script compares theoretical X-Ray Diffraction (XRD) patterns generated from `.cif` files to determine if two structures represent the same polymorph. 

It evaluates the match using the de Gelder cross-correlation function and automatically applies an angular shift (2θ) to correct for minor unit cell volume differences, which typically occur due to thermal expansion or DFT relaxation artifacts.

## Dependencies

You will need Python 3 and a few standard libraries. You can install everything you need via pip:

```bash
pip install numpy scipy matplotlib pymatgen

```

## Usage
There is no need to pass command-line arguments. To configure your run, open the script and scroll down to the main() function. You will find a configuration block that looks like this:

```bash
REFERENCE_CIF = "Quercetin/NAFZEC02.cif"
ANALYSIS_MODE = "batch"  # or "single"
CANDIDATE_FILE = "Quercetin/Struttura_Deep_Relax.cif"
CANDIDATE_FOLDER = "Quercetin/Top50_Best_Refined" 
Set your REFERENCE_CIF path.
```

Choose the ANALYSIS_MODE:

Set it to "single" to compare one structure (update CANDIDATE_FILE).

Set it to "batch" to process an entire directory of CIFs against the reference (update CANDIDATE_FOLDER).

Run the script from your terminal:

```bash
python comparison_script.py
```

## Outputs
The script creates an output folder automatically. For each candidate, it generates:

A .png plot showing the reference, the original candidate, and the optimally shifted candidate.

Raw .xy text files (2θ vs Intensity) for all patterns, easily importable into OriginLab or Excel.

A Summary_Report.txt containing the exact similarity score, dissimilarity score, and applied shift for every file.

Note on batch mode: To avoid flooding the terminal when analyzing dozens of files, the script will only print the Top 5 best matches and the single worst candidate to standard output. The full results are always safely stored in the Summary_Report.txt.

Rule of Thumb: As a general rule from literature, a dissimilarity score < 0.03 indicates that the two structures share the same polymorph framework.

Peak Resolution & Advanced Tweaking
If you need to adjust the simulated X-ray source (default is Cu K-alpha: 1.54056 Å) or the 2θ range, you can modify the global variables at the top of the script.

The script currently uses standard empirical Caglioti parameters to simulate the peak broadening of a conventional lab diffractometer. If you need sharper peaks (e.g., simulating synchrotron radiation or perfect crystals), locate these variables inside the generate_clean_pattern function:

```bash
u, v, w = 0.04, -0.004, 0.04
```

Lowering these values (for instance to 0.001, -0.001, 0.001) will narrow the FWHM of the peaks accordingly.
