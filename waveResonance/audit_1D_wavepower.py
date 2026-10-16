import os
import sys
import glob
import argparse
import cdflib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

def check_cdf_file(filepath, var_name="integrated_wave_power"):
    """
    Inspects a single CDF file to check if a specified variable 
    is entirely filled with dummy/fill values or NaN.
    """
    try:
        cdf = cdflib.CDF(filepath)
        info = cdf.cdf_info()
        
        # Get variable list (handles dict or object attribute formats across cdflib versions)
        zvars = info.zVariables if hasattr(info, 'zVariables') else info['zVariables']
        
        # Case-insensitive variable lookup
        target_var = None
        for v in zvars:
            if v.lower() == var_name.lower():
                target_var = v
                break
                
        if not target_var:
            return {
                'file': filepath,
                'status': 'MISSING_VAR',
                'n_records': 0,
                'msg': f"Variable '{var_name}' not found in file"
            }
            
        data = cdf.varget(target_var)
        if data is None or len(data) == 0:
            return {
                'file': filepath,
                'status': 'EMPTY_VAR',
                'n_records': 0,
                'msg': "Variable is empty"
            }
            
        data = np.asarray(data)
        n_records = len(data)
        
        # CDF fill values are typically <= -1e30
        valid_mask = (data > -1e30) & np.isfinite(data)
        n_valid = np.sum(valid_mask)
        
        if n_valid == 0:
            return {
                'file': filepath,
                'status': 'ALL_FILL',
                'n_records': n_records,
                'msg': f"All {n_records} records are CDF fill/NaN"
            }
        else:
            return {
                'file': filepath,
                'status': 'OK',
                'n_records': n_records,
                'n_valid': n_valid,
                'msg': f"{n_valid}/{n_records} valid records ({n_valid/n_records*100:.1f}%)"
            }
            
    except Exception as e:
        return {
            'file': filepath,
            'status': 'CORRUPT_OR_ERROR',
            'n_records': 0,
            'msg': f"CDF Read Error: {e}"
        }

def main():
    parser = argparse.ArgumentParser(description="Audit 1D WaveResonance CDF Files for Fill/NaN Variables")
    parser.add_argument("--root", type=str, default="data/cdf_files/PSP_Waves", help="Root directory containing CDF files")
    parser.add_argument("--var", type=str, default="integrated_wave_power", help="Target variable name to audit")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes")
    parser.add_argument("--output", type=str, default="missing_wavepower_files.txt", help="Text file to save list of ALL_FILL files")
    args = parser.parse_args()

    if not os.path.exists(args.root):
        print(f"[!] Error: Root directory '{args.root}' does not exist.")
        sys.exit(1)

    print(f"[*] Recursively searching for CDF files under: {args.root}")
    # glob pattern matching both standard directory names and trailing-space directories
    pattern = os.path.join(args.root, "**", "PSP_WaveResonance_1D_*.cdf")
    cdf_files = sorted(glob.glob(pattern, recursive=True))

    if not cdf_files:
        print("[!] No CDF files matching 'PSP_WaveResonance_1D_*.cdf' found.")
        sys.exit(0)

    print(f"[*] Found {len(cdf_files)} CDF files. Auditing variable '{args.var}' across {args.workers} workers...")

    results = []
    all_fill_files = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {executor.submit(check_cdf_file, f, args.var): f for f in cdf_files}
        
        for future in as_completed(future_to_file):
            res = future.result()
            results.append(res)
            if res['status'] == 'ALL_FILL':
                all_fill_files.append(res['file'])

    # Categorize results
    ok_count = sum(1 for r in results if r['status'] == 'OK')
    fill_count = sum(1 for r in results if r['status'] == 'ALL_FILL')
    missing_count = sum(1 for r in results if r['status'] == 'MISSING_VAR')
    err_count = sum(1 for r in results if r['status'] == 'CORRUPT_OR_ERROR')

    print("\n" + "=" * 70)
    print(f"AUDIT SUMMARY FOR VARIABLE: '{args.var}'")
    print("=" * 70)
    print(f"Total Files Scanned : {len(results)}")
    print(f"  [+] Valid Data    : {ok_count} ({ok_count/len(results)*100:.1f}%)")
    print(f"  [-] ALL FILL / NaN : {fill_count} ({fill_count/len(results)*100:.1f}%)")
    print(f"  [!] Variable Missing: {missing_count}")
    print(f"  [!] Errors/Corrupt: {err_count}")
    print("=" * 70)

    if all_fill_files:
        all_fill_files.sort()
        with open(args.output, "w") as out_f:
            for f in all_fill_files:
                out_f.write(f"{f}\n")
        print(f"\n[+] Saved list of {len(all_fill_files)} problematic files to: {args.output}")

if __name__ == "__main__":
    main()