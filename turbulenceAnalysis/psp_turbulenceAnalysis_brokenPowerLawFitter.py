#!/usr/bin/env python3
"""
psp_turbulenceAnalysis_brokenPowerLawFitter.py
=============================================
Multiprocess batch extraction of turbulent magnetic spectral breaks, cascade slopes, 
and coherent ion-scale wave excess power for Parker Solar Probe (FIELDS/SWEAP).

Implements the concurrent worker queue, Rich live dashboard, and terminal notifications
from psp_waveResonance.py while maintaining platform-agnostic execution across CFA servers
(fc.cfa.harvard.edu) and local Windows/macOS workstations.

Pipeline Mapping:
    Input:  DRIVE_ROOT/Research/PSP/WaveAnalysis/WaveAnalysis_Files/v1.X/YYYY/MM/
    Output: DRIVE_ROOT/Research/PSP/TurbulenceAnalysis/TurbulenceAnalysis_Files/v2.X/YYYY/MM/
"""

import os
import sys
import glob
import gc
import socket
import platform
import argparse
import tempfile
import shutil
import time
import warnings
import concurrent.futures
from datetime import datetime, timedelta, timezone
from multiprocessing import Manager
from contextlib import redirect_stdout

# Cross-platform memory guard import (Unix only; gracefully skipped on Windows)
try:
    import resource
except ImportError:
    resource = None

# Rich terminal dashboard imports
from rich.live import Live
from rich.table import Table
from rich.panel import Panel

import numpy as np
import pandas as pd
import cdflib
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Look up one level to discover central shared repository modules (config.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    import config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# =============================================================================
# PUBLICATION PLOTTING CONFIGURATION (AGU / GRL / ApJ STANDARDS)
# =============================================================================
# Strict academic print standards: heavy spines, inward ticks, >=12pt text
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.5,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'xtick.major.width': 1.5,
    'ytick.major.size': 6,
    'ytick.major.width': 1.5,
    'xtick.minor.size': 3.5,
    'xtick.minor.width': 1.0,
    'ytick.minor.size': 3.5,
    'ytick.minor.width': 1.0,
    'lines.linewidth': 1.6,
    'figure.autolayout': True
})

# High-contrast, colorblind-safe Okabe-Ito palette
PALETTE = {
    'turb_raw':    '#999999',  # Incoherent point scatter
    'turb_median': '#000000',  # Frequency-binned turbulent continuum
    'wave_median': '#D55E00',  # Coherent wave emission
    'model_fit':   '#0072B2',  # Smooth broken power law model
    'break_freq':  '#CC79A7',  # Spectral break locator
    'excess_fill': '#F0E442'   # Integrated wave excess power fill
}

def log_status(message: str) -> None:
    """Outputs execution logs with formatted time-stamps [HH:MM:SS]."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}", flush=True)

# Compatibility shim for NumPy trapezoidal integration across NumPy 1.x and 2.x
_trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

# =============================================================================
# MULTIPROCESSING QUEUE LOGGER & SYSTEM CONTROLS
# =============================================================================
class QueueLogger:
    """Redirects print statements from workers directly to the Rich console queue."""
    def __init__(self, queue):
        self.queue = queue
    def write(self, msg: str):
        msg_str = msg.strip()
        if msg_str:
            self.queue.put(msg_str)
    def flush(self):
        pass

def limit_memory(max_gb: float = 8.0) -> None:
    """Prevents any single process worker from exceeding memory limits on POSIX systems."""
    if resource is None:
        return
    try:
        max_bytes = int(max_gb * 1024 * 1024 * 1024)
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))
    except Exception:
        pass

def get_peak_ram_str() -> str:
    """Retrieves current resident memory (RSS) across Linux, macOS, and Windows."""
    # POSIX / Linux / CFA server
    if resource is not None:
        try:
            peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if platform.system() == 'Darwin':
                peak_kb /= 1024.0
            return f"{peak_kb / (1024.0 * 1024.0):.2f} GB"
        except Exception:
            pass

    # Windows native kernel call via standard library ctypes
    if platform.system() == 'Windows':
        try:
            import ctypes
            from ctypes import wintypes
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ('cb', wintypes.DWORD),
                    ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t),
                ]
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return f"{counters.WorkingSetSize / (1024.0**3):.2f} GB"
        except Exception:
            pass

    return "-"

def set_terminal_tab_title(title_text: str) -> None:
    """Writes OSC escape sequences directly to controlling terminal devices."""
    try:
        if os.name == 'posix':
            with open('/dev/tty', 'w') as tty:
                tty.write(f"\033]0;{title_text}\007")
                tty.flush()
        elif os.name == 'nt':
            with open('CON', 'w') as tty:
                tty.write(f"\033]0;{title_text}\007")
                tty.flush()
    except Exception:
        pass

def make_dashboard(worker_states: dict, completed: int, total: int) -> Panel:
    """Renders a dynamic live multi-row UI panel for turbulence fitting workers."""
    table = Table(expand=True)
    table.add_column("Worker", justify="center", style="cyan bold", no_wrap=True)
    table.add_column("Date", justify="center", style="green")
    table.add_column("Chunk", justify="center", style="yellow")
    table.add_column("Status", justify="left", style="white")
    table.add_column("Peak RAM", justify="right", style="magenta")

    for slot_id in sorted(worker_states.keys()):
        state = worker_states[slot_id]
        table.add_row(
            f"Worker {slot_id + 1}",
            state.get('date', 'Idle'),
            state.get('chunk', '-'),
            state.get('status', 'Idle'),
            state.get('ram', '-')
        )
        
    pct = (completed / total * 100) if total > 0 else 0.0
    title_str = f"PSP Turbulence Spectral Fitter Pipeline | Progress: {completed}/{total} Days ({pct:.1f}%)"
    return Panel(table, title=title_str, border_style="bold cyan")

# =============================================================================
# ENVIRONMENT & PATH RESOLUTION
# =============================================================================
def resolve_drive_root() -> str:
    """Detects runtime host and resolves Google Drive research directory root."""
    if CONFIG_AVAILABLE:
        try:
            return config.get_drive_path().rstrip('/\\')
        except Exception:
            pass

    hostname = socket.gethostname().lower()
    if 'fc.cfa.harvard.edu' in hostname or hostname.startswith('fc'):
        return '/home/kpaulson/MyDrive'

    if platform.system() == 'Windows':
        for candidate in ['G:/My Drive', 'G:\\My Drive']:
            if os.path.exists(candidate):
                return candidate
        return 'G:/My Drive'

    expanded_mydrive = os.path.expanduser('~/MyDrive')
    if os.path.exists(expanded_mydrive):
        return expanded_mydrive

    return os.path.expanduser('~')

def derive_output_version(v_in: str) -> str:
    """Constructs output turbulence version 2.X for input waveAnalysis version 1.X."""
    minor_parts = v_in.split('.', 1)[-1]
    return f"2.{minor_parts}"

# =============================================================================
# BLOCK 1: DATA LOADING & LOCAL RAM CACHING
# =============================================================================
def load_wave_analysis_cdf(cdf_path: str) -> dict:
    """Directly extracts spectral matrices, frequencies, and coherency using a local temp copy."""
    if not os.path.exists(cdf_path):
        raise FileNotFoundError(f"WaveAnalysis CDF file not found: {cdf_path}")

    # Stage locally in /tmp to prevent network file-lock freezes on shared mounts
    fd, temp_file_path = tempfile.mkstemp(suffix='.cdf')
    os.close(fd)
    shutil.copyfile(cdf_path, temp_file_path)

    wave_dict = {}
    try:
        with cdflib.CDF(temp_file_path) as cdf:
            # TT2000 epoch conversion to POSIX seconds
            raw_time = cdf.varget('fft_time')
            wave_dict['unix_time_s'] = cdflib.cdfepoch.unixtime(raw_time)

            # Frequency axis in spacecraft frame (Hz)
            freqs = cdf.varget('frequencies')
            if freqs.ndim > 1:
                freqs = freqs[0, :]
            wave_dict['frequencies_hz'] = np.ascontiguousarray(freqs, dtype=np.float64)

            # 2D Spectrograms: [Time x Frequency]
            power_b = cdf.varget('wave_power_B')
            coherency = cdf.varget('coherency')

            # Physical cleaning:
            # 1. Mask non-positive power values and standard CDF fill flags (-1e30)
            # 2. Coherency is bounded strictly between 0 and 1
            power_b = np.where((power_b <= 1e-10) | (power_b < -1e30), np.nan, power_b)
            coherency = np.where((coherency < 0.0) | (coherency > 1.0), np.nan, coherency)

            wave_dict['power_b_nt2_per_hz'] = np.ascontiguousarray(power_b, dtype=np.float64)
            wave_dict['coherency_unitless'] = np.ascontiguousarray(coherency, dtype=np.float64)
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass

    return wave_dict

# =============================================================================
# BLOCK 2: PHYSICAL MODELS & CONTINUUM FITTING
# =============================================================================
def smooth_broken_power_law(
    log10_f: np.ndarray, 
    log10_fb: float, 
    log10_p0: float, 
    alpha_inertial: float, 
    alpha_kinetic: float, 
    delta: float
) -> np.ndarray:
    """Continuously differentiable broken power law in log10 space."""
    x = log10_f - log10_fb
    # Clip exponential argument to [-50, 50] to eliminate floating-point overflow
    arg = np.clip(x / delta, -50.0, 50.0)
    transition = delta * np.log10(0.5 * (1.0 + 10.0**arg))
    return log10_p0 - alpha_inertial * x + (alpha_inertial - alpha_kinetic) * transition

def fit_turbulent_continuum(
    frequencies_hz: np.ndarray, 
    median_power_turb_nt2_per_hz: np.ndarray
) -> dict:
    """Fits the smooth broken power law to the incoherent turbulent continuum using TRF."""
    valid_mask = (
        np.isfinite(median_power_turb_nt2_per_hz) & 
        (median_power_turb_nt2_per_hz > 0.0) & 
        (frequencies_hz > 0.0)
    )
    if np.count_nonzero(valid_mask) < 12:
        return None

    x_data = np.log10(frequencies_hz[valid_mask])
    y_data = np.log10(median_power_turb_nt2_per_hz[valid_mask])

    # Parameter bounds: [log10_fb, log10_p0, alpha_inertial, alpha_kinetic, delta]
    param_bounds = (
        [-1.5, -8.0, 0.5, 1.8, 0.02],
        [ 2.0,  5.0, 2.5, 6.0, 0.80]
    )

    idx_1hz = np.argmin(np.abs(x_data - 0.0))
    p0_guess = y_data[idx_1hz] if len(y_data) > idx_1hz else -1.0
    init_guesses = [0.0, p0_guess, 1.67, 3.2, 0.20]

    try:
        popt, pcov = curve_fit(
            smooth_broken_power_law, 
            x_data, 
            y_data, 
            p0=init_guesses, 
            bounds=param_bounds,
            maxfev=2000,
            method='trf'
        )

        residuals = y_data - smooth_broken_power_law(x_data, *popt)
        rmse_log_dex = np.sqrt(np.mean(residuals**2))

        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(5, np.nan)
        fb_val = 10.0**popt[0]
        sigma_fb = np.log(10.0) * fb_val * perr[0]

        return {
            'log10_fb': popt[0],
            'fb_hz': fb_val,
            'sigma_fb_hz': sigma_fb,
            'log10_p0': popt[1],
            'p0_nt2_per_hz': 10.0**popt[1],
            'alpha_inertial': popt[2],
            'sigma_alpha_inertial': perr[2],
            'alpha_kinetic': popt[3],
            'sigma_alpha_kinetic': perr[3],
            'delta_decades': popt[4],
            'rmse_log_dex': rmse_log_dex,
            'fit_success': True
        }
    except Exception:
        return None

# =============================================================================
# BLOCK 3: WINDOWED SPECTRAL PROCESSING & INTEGRATION
# =============================================================================
def process_spectral_window(
    frequencies_hz: np.ndarray, 
    power_window_nt2_per_hz: np.ndarray, 
    coherency_window: np.ndarray,
    coh_turb_threshold: float = 0.5, 
    coh_wave_threshold: float = 0.8
) -> dict:
    """Separates incoherent turbulence from coherent waves and extracts power metrics."""
    # 1. Mask incoherent background (coherency <= 0.5: random turbulent cascade)
    incoherent_mask = (coherency_window <= coh_turb_threshold) & np.isfinite(power_window_nt2_per_hz)
    turb_power_slice = np.where(incoherent_mask, power_window_nt2_per_hz, np.nan)

    # Suppress RuntimeWarning for frequency channels that are 100% masked in this slice
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        log_turb_median = np.nanmedian(np.log10(turb_power_slice), axis=0)
        median_power_turb = 10.0**log_turb_median

    # 2. Fit smooth broken power law
    fit_metrics = fit_turbulent_continuum(frequencies_hz, median_power_turb)
    if fit_metrics is None:
        return None

    fb_hz = fit_metrics['fb_hz']
    model_log_power = smooth_broken_power_law(
        np.log10(frequencies_hz),
        fit_metrics['log10_fb'],
        fit_metrics['log10_p0'],
        fit_metrics['alpha_inertial'],
        fit_metrics['alpha_kinetic'],
        fit_metrics['delta_decades']
    )
    model_power_nt2_per_hz = 10.0**model_log_power

    # 3. Integrate turbulent powers across distinct physical regimes (nT^2)
    turb_power_total_nt2 = _trapezoid(model_power_nt2_per_hz, frequencies_hz)

    inertial_mask = frequencies_hz <= fb_hz
    kinetic_mask = frequencies_hz >= fb_hz
    turb_power_inertial_nt2 = _trapezoid(model_power_nt2_per_hz[inertial_mask], frequencies_hz[inertial_mask])
    turb_power_kinetic_nt2 = _trapezoid(model_power_nt2_per_hz[kinetic_mask], frequencies_hz[kinetic_mask])

    # 4. Extract coherent wave packets (coherency >= 0.8: polarized wave modes)
    coherent_mask = (coherency_window >= coh_wave_threshold) & np.isfinite(power_window_nt2_per_hz)
    wave_power_slice = np.where(coherent_mask, power_window_nt2_per_hz, np.nan)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        log_wave_median = np.nanmedian(np.log10(wave_power_slice), axis=0)
        median_power_wave = 10.0**log_wave_median

    # 5. Continuum-subtracted excess power: delta B_wave^2 = integral max(0, P_wave - P_fit) df
    wave_excess = np.maximum(0.0, median_power_wave - model_power_nt2_per_hz)
    wave_excess_clean = np.where(np.isfinite(wave_excess), wave_excess, 0.0)
    wave_excess_power_nt2 = _trapezoid(wave_excess_clean, frequencies_hz)

    if np.any(wave_excess_clean > 0.0):
        peak_idx = np.argmax(wave_excess_clean)
        peak_wave_freq_hz = frequencies_hz[peak_idx]
        peak_wave_density_nt2_per_hz = median_power_wave[peak_idx]
    else:
        peak_wave_freq_hz = np.nan
        peak_wave_density_nt2_per_hz = np.nan

    wave_to_turb_ratio = (
        wave_excess_power_nt2 / turb_power_total_nt2 
        if turb_power_total_nt2 > 0.0 else np.nan
    )

    return {
        **fit_metrics,
        'turb_power_total_nt2': turb_power_total_nt2,
        'turb_power_inertial_nt2': turb_power_inertial_nt2,
        'turb_power_kinetic_nt2': turb_power_kinetic_nt2,
        'wave_excess_power_nt2': wave_excess_power_nt2,
        'peak_wave_freq_hz': peak_wave_freq_hz,
        'peak_wave_density_nt2_per_hz': peak_wave_density_nt2_per_hz,
        'wave_to_turb_ratio': wave_to_turb_ratio,
        'frequencies_hz': frequencies_hz,
        'median_power_turb_nt2_per_hz': median_power_turb,
        'median_power_wave_nt2_per_hz': median_power_wave,
        'model_power_nt2_per_hz': model_power_nt2_per_hz,
        'wave_excess_clean_nt2_per_hz': wave_excess_clean
    }

# =============================================================================
# BLOCK 4: PUBLICATION-READY VISUALIZATION (AGU / GRL FORMATTING)
# =============================================================================
def plot_diagnostic_spectral_fit(
    metrics: dict, 
    mid_dt_utc: datetime, 
    output_png_path: str
) -> None:
    """Renders an AGU/GRL journal-standard diagnostic fit with LaTeX units and raw strings."""
    freqs = metrics['frequencies_hz']
    turb_data = metrics['median_power_turb_nt2_per_hz']
    wave_data = metrics['median_power_wave_nt2_per_hz']
    model_power = metrics['model_power_nt2_per_hz']
    fb_hz = metrics['fb_hz']

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    ax.scatter(
        freqs, turb_data, color=PALETTE['turb_median'], s=22, alpha=0.9,
        edgecolors='none', label=r'Incoherent Continuum ($C \leq 0.5$)', zorder=3
    )

    valid_wave = np.isfinite(wave_data) & (wave_data > 0)
    if np.any(valid_wave):
        ax.scatter(
            freqs[valid_wave], wave_data[valid_wave], color=PALETTE['wave_median'], 
            s=26, marker='^', alpha=0.95, edgecolors='none',
            label=r'Coherent Wave Modes ($C \geq 0.8$)', zorder=4
        )
        
        ax.fill_between(
            freqs, model_power, np.maximum(model_power, wave_data),
            where=(wave_data > model_power), color=PALETTE['excess_fill'],
            alpha=0.45, label=r'Wave Excess Power ($\delta B_{\mathrm{wave}}^2$)', zorder=2
        )

    ax.plot(
        freqs, model_power, color=PALETTE['model_fit'], linewidth=2.0,
        label=rf'BPL Model ($\delta={metrics["delta_decades"]:.2f}\ \mathrm{{dex}}$)', zorder=5
    )

    ax.axvline(
        fb_hz, color=PALETTE['break_freq'], linestyle='--', linewidth=1.5,
        label=rf'Break $f_b = {fb_hz:.2f}\ \mathrm{{Hz}}$', zorder=2
    )

    summary_lines = [
        r"$\mathbf{Spectral\ Parameters:}$",
        rf"$\alpha_{{\mathrm{{inertial}}}} = {metrics['alpha_inertial']:.2f} \pm {metrics['sigma_alpha_inertial']:.2f}$",
        rf"$\alpha_{{\mathrm{{kinetic}}}} = {metrics['alpha_kinetic']:.2f} \pm {metrics['sigma_alpha_kinetic']:.2f}$",
        rf"$P_{{\mathrm{{turb,tot}}}} = {metrics['turb_power_total_nt2']:.2e}\ \mathrm{{nT}}^2$",
        rf"$\delta B_{{\mathrm{{wave}}}}^2 = {metrics['wave_excess_power_nt2']:.2e}\ \mathrm{{nT}}^2$",
        rf"$\mathrm{{RMSE}} = {metrics['rmse_log_dex']:.3f}\ \mathrm{{dex}}$"
    ]
    summary_text = "\n".join(summary_lines)
    ax.text(
        0.04, 0.05, summary_text, transform=ax.transAxes, fontsize=10.5,
        verticalalignment='bottom', horizontalalignment='left',
        bbox=dict(boxstyle='square,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.92)
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Spacecraft Frame Frequency $f_{\mathrm{sc}}\ (\mathrm{Hz})$')
    ax.set_ylabel(r'Trace Magnetic Power Density $P_B\ (\mathrm{nT}^2/\mathrm{Hz})$')
    ax.set_title(f'PSP FIELDS Magnetic Spectrum | {mid_dt_utc.strftime("%Y-%m-%d %H:%M:%S UTC")}')
    ax.grid(True, which='both', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.legend(loc='upper right', frameon=False, fontsize=9.5)

    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    gc.collect()

# =============================================================================
# BLOCK 5: MULTIPROCESS WORKER TASK
# =============================================================================
def process_day_worker(day_tuple, slot_queue, status_queue, log_queue):
    """
    Worker task: Processes 1 full observation day (all 4 chunks) with live UI telemetry.
    """
    current_date, args, base_dir, output_dir, v_in, v_out = day_tuple
    limit_memory(8.0) # Restrict worker RSS to 8 GB max on Linux

    slot_id = slot_queue.get() # Acquire dashboard slot
    y_str = current_date.strftime('%Y')
    m_str = current_date.strftime('%m')
    d_str = current_date.strftime('%d')
    date_str = f"{y_str}-{m_str}-{d_str}"

    day_out_dir = os.path.join(output_dir, y_str, m_str)
    plot_out_dir = os.path.join(day_out_dir, 'plots')
    out_csv_path = os.path.join(day_out_dir, f"PSP_TurbulenceAnalysis_BPL_{date_str}_v{v_out}.csv")

    def update_ui(status_text: str, chunk_str: str = "-"):
        """Pushes worker state to main rendering thread."""
        status_queue.put({
            'slot_id': slot_id, 'date': date_str, 'chunk': chunk_str,
            'status': status_text, 'ram': get_peak_ram_str()
        })

    out_stream = QueueLogger(log_queue) if args.verbose else open(os.devnull, 'w')

    try:
        with redirect_stdout(out_stream):
            update_ui("Checking Overwrite", "-")

            # Check if day is already completed
            if not args.overwrite and os.path.exists(out_csv_path):
                if args.verbose:
                    print(f"[*] Skipping Day: {date_str} (CSV Output Exists)")
                update_ui("Skipped (Exists)", "-")
                return True

            if args.verbose:
                print(f"[=========== Processing Day: {date_str} (v{v_in} -> v{v_out}) ===========]")

            os.makedirs(day_out_dir, exist_ok=True)
            if args.save_plots:
                os.makedirs(plot_out_dir, exist_ok=True)

            # Determine daily input directory (with Nanook fallback on CFA server)
            day_in_dir = os.path.join(base_dir, y_str, m_str)
            nanook_in_dir = os.path.join('/mnt', 'nanook', 'waveAnalysis_PSP', 'WaveAnalysis_Files', f'v{v_in}', y_str, m_str)
            if not os.path.exists(day_in_dir) and os.path.exists(nanook_in_dir):
                day_in_dir = nanook_in_dir

            daily_results = []

            for hr in [0, 6, 12, 18]:
                hr_tag = f"{hr:02d}00"
                chunk_str = f"{hr:02d}:00"
                update_ui("Resolving Files", chunk_str)

                search_pattern = os.path.join(day_in_dir, f"PSP_WaveAnalysis_{y_str}-{m_str}-{d_str}_{hr_tag}_v{v_in}.cdf")
                matches = sorted(glob.glob(search_pattern))

                if not matches:
                    wildcard_pattern = os.path.join(day_in_dir, f"PSP_WaveAnalysis_{y_str}-{m_str}-{d_str}_{hr_tag}_v*.cdf")
                    matches = sorted(glob.glob(wildcard_pattern))

                if not matches:
                    if args.verbose:
                        print(f"  [!] Chunk {chunk_str} -> CDF missing. Skipping.")
                    continue

                active_cdf = matches[-1]
                update_ui("Reading CDF", chunk_str)

                try:
                    wave_telemetry = load_wave_analysis_cdf(active_cdf)
                except Exception as e:
                    if args.verbose:
                        print(f"  [!] Read failure on {os.path.basename(active_cdf)}: {e}")
                    continue

                timestamps_s = wave_telemetry['unix_time_s']
                frequencies_hz = wave_telemetry['frequencies_hz']
                power_matrix = wave_telemetry['power_b_nt2_per_hz']
                coherency_matrix = wave_telemetry['coherency_unitless']

                total_slices = len(timestamps_s)
                total_windows = max(1, (total_slices - args.step) // args.step + 1)
                window_start_idx = 0
                window_idx = 0

                while window_start_idx + args.step <= total_slices:
                    window_idx += 1
                    
                    # Heartbeat update to live dashboard every 5 windows
                    if window_idx % 5 == 0 or window_idx == total_windows:
                        pct_chunk = int((window_idx / total_windows) * 100)
                        update_ui(f"Fit {window_idx}/{total_windows} ({pct_chunk}%)", chunk_str)

                    power_slice = power_matrix[window_start_idx:window_start_idx + args.step, :]
                    coherency_slice = coherency_matrix[window_start_idx:window_start_idx + args.step, :]
                    
                    midpoint_idx = window_start_idx + (args.step // 2)
                    mid_time_s = timestamps_s[midpoint_idx]
                    mid_dt_utc = datetime.fromtimestamp(mid_time_s, tz=timezone.utc)

                    metrics = process_spectral_window(frequencies_hz, power_slice, coherency_slice)

                    if metrics is not None:
                        record = {
                            'time_utc': mid_dt_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                            'epoch_unix_s': mid_time_s,
                            'break_freq_hz': metrics['fb_hz'],
                            'break_freq_sigma_hz': metrics['sigma_fb_hz'],
                            'power_density_at_break_nt2_per_hz': metrics['p0_nt2_per_hz'],
                            'alpha_inertial_slope': metrics['alpha_inertial'],
                            'alpha_inertial_sigma': metrics['sigma_alpha_inertial'],
                            'alpha_kinetic_slope': metrics['alpha_kinetic'],
                            'alpha_kinetic_sigma': metrics['sigma_alpha_kinetic'],
                            'transition_delta_decades': metrics['delta_decades'],
                            'turbulent_power_total_nt2': metrics['turb_power_total_nt2'],
                            'turbulent_power_inertial_nt2': metrics['turb_power_inertial_nt2'],
                            'turbulent_power_kinetic_nt2': metrics['turb_power_kinetic_nt2'],
                            'wave_excess_power_nt2': metrics['wave_excess_power_nt2'],
                            'peak_wave_freq_hz': metrics['peak_wave_freq_hz'],
                            'peak_wave_density_nt2_per_hz': metrics['peak_wave_density_nt2_per_hz'],
                            'wave_to_turb_energy_ratio': metrics['wave_to_turb_ratio'],
                            'fit_rmse_log_dex': metrics['rmse_log_dex']
                        }
                        daily_results.append(record)

                        if args.save_plots:
                            tag_str = mid_dt_utc.strftime('%Y%m%d_%H%M%S')
                            plot_filename = f"PSP_TurbulenceAnalysis_BPL_{tag_str}_v{v_out}.png"
                            plot_path = os.path.join(plot_out_dir, plot_filename)
                            plot_diagnostic_spectral_fit(metrics, mid_dt_utc, plot_path)

                    window_start_idx += args.step

                del wave_telemetry
                gc.collect()

            if daily_results:
                update_ui("Writing CSV", "Daily Merge")
                df_out = pd.DataFrame(daily_results)
                df_out.to_csv(out_csv_path, index=False)
                if args.verbose:
                    print(f"  [+] Wrote {len(df_out)} fits to {os.path.basename(out_csv_path)}")
                update_ui("Complete", "Done")
            else:
                update_ui("No Valid Fits", "Done")

            return True

    except Exception as e:
        update_ui(f"Failed: {e}", "-")
        log_queue.put(f"  [!] Fatal failure on {date_str}: {e}")
        return False

    finally:
        if not args.verbose:
            out_stream.close()
        gc.collect()
        slot_queue.put(slot_id) # Release worker slot

# =============================================================================
# BLOCK 6: MULTIPROCESS BATCH SCHEDULER & DASHBOARD LOOP
# =============================================================================
def main():
    args = setup_cli()

    drive_root = resolve_drive_root()
    v_in = args.v_in
    v_out = derive_output_version(v_in)

    if args.base_dir:
        base_dir = args.base_dir
    else:
        base_dir = os.path.join(drive_root, 'Research', 'PSP', 'WaveAnalysis', 'WaveAnalysis_Files', f'v{v_in}')

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(drive_root, 'Research', 'PSP', 'TurbulenceAnalysis', 'TurbulenceAnalysis_Files', f'v{v_out}')

    # Parse date or range syntax
    try:
        if ' through ' in args.date:
            date_parts = args.date.split(' through ')
            start_date = datetime.strptime(date_parts[0].strip(), '%Y-%m-%d')
            end_date = datetime.strptime(date_parts[1].strip(), '%Y-%m-%d')
        else:
            start_date = datetime.strptime(args.date.strip(), '%Y-%m-%d')
            end_date = start_date
    except Exception as e:
        print(f"[!] Error parsing --date argument: {e}")
        return

    # Build master list of daily tasks
    day_tasks = []
    curr = start_date
    while curr <= end_date:
        task_tuple = (curr, args, base_dir, output_dir, v_in, v_out)
        day_tasks.append(task_tuple)
        curr += timedelta(days=1)

    total_days = len(day_tasks)
    completed_days = 0

    print(f"[*] Commencing Multiprocess Turbulence Fitting: {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")
    print(f"[*] Hostname: {socket.gethostname()} | Workers: {args.workers}")
    print(f"[*] DRIVE_ROOT: {drive_root}")
    print(f"[*] In: {base_dir} (v{v_in}) -> Out: {output_dir} (v{v_out})\n")

    # Multiprocessing synchronization queues and slot allocations
    manager = Manager()
    slot_queue = manager.Queue()
    status_queue = manager.Queue()
    log_queue = manager.Queue()

    for i in range(args.workers):
        slot_queue.put(i)

    worker_states = {i: {'date': 'Idle', 'chunk': '-', 'status': 'Idle', 'ram': '-'} for i in range(args.workers)}

    # Process pool execution with Rich live dashboard rendering
    try:
        with Live(make_dashboard(worker_states, 0, total_days), refresh_per_second=4) as live:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_day_worker, task, slot_queue, status_queue, log_queue): task
                    for task in day_tasks
                }

                completed_futures = set()
                last_completed = -1

                while completed_days < total_days:
                    # 1. Print worker logs safely above dashboard
                    while not log_queue.empty():
                        msg = log_queue.get()
                        live.console.print(msg)

                    # 2. Update worker table state
                    while not status_queue.empty():
                        msg = status_queue.get()
                        slot = msg['slot_id']
                        worker_states[slot].update(msg)

                    # 3. Harvest completed tasks
                    done = [f for f in futures if f.done() and f not in completed_futures]
                    for f in done:
                        completed_futures.add(f)
                        completed_days += 1

                    # 4. Tabby terminal tab update
                    if completed_days != last_completed:
                        last_completed = completed_days
                        pct = int((completed_days / total_days) * 100)
                        set_terminal_tab_title(f"[{pct}%] TurbBPL ({completed_days}/{total_days} Days)")

                    live.update(make_dashboard(worker_states, completed_days, total_days))
                    time.sleep(0.1)

    except KeyboardInterrupt:
        set_terminal_tab_title("[ABORTED] TurbBPL")
        print("\n\n[!] KeyboardInterrupt detected. Terminating worker pool...")
        executor.shutdown(wait=False, cancel_futures=True)
        manager.shutdown()
        sys.exit(1)

    # Terminal completion sound and notification
    try:
        if os.name == 'posix':
            with open('/dev/tty', 'w') as tty:
                tty.write("\033]0;[DONE] TurbBPL Complete!\007")
                tty.write("\033]9;Parker Solar Probe Turbulence BPL fitting complete!\007")
                tty.flush()
        elif os.name == 'nt':
            with open('CON', 'w') as tty:
                tty.write("\033]0;[DONE] TurbBPL Complete!\007")
                tty.write("\033]9;Parker Solar Probe Turbulence BPL fitting complete!\007")
                tty.flush()
    except Exception:
        pass

    print("\n[+] Multiprocessing turbulence broken power law fitting complete.")

# =============================================================================
# CLI ROUTING
# =============================================================================
def setup_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multiprocess BPL Turbulence Extraction from PSP WaveAnalysis CDFs."
    )
    parser.add_argument(
        '--date', type=str, required=True, 
        help="Target date or range: 'YYYY-MM-DD' or 'YYYY-MM-DD through YYYY-MM-DD'"
    )
    parser.add_argument(
        '-v_in', '--v_in', type=str, default="1.4", 
        help="WaveAnalysis input version (default: '1.4'). Output version is mapped as 2.X."
    )
    parser.add_argument(
        '--workers', type=int, default=2, 
        help="Number of daily worker processes to run concurrently (default: 2)"
    )
    parser.add_argument(
        '--step', type=int, default=50, 
        help="Sliding window size (number of spectra) per spectral fit (default: 50)"
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', 
        help="Print detailed worker progress messages"
    )
    parser.add_argument(
        '--overwrite', '-o', action='store_true', 
        help="Overwrite existing daily CSV output files"
    )
    parser.add_argument(
        '--save-plots', action='store_true', 
        help="Generate AGU publication-standard diagnostic PNGs for each fit"
    )
    parser.add_argument(
        '--base-dir', type=str, default=None, 
        help="Override base path for WaveAnalysis CDF files"
    )
    parser.add_argument(
        '--output-dir', type=str, default=None, 
        help="Override root directory for output CSVs and plots"
    )
    return parser.parse_args()

if __name__ == '__main__':
    main()