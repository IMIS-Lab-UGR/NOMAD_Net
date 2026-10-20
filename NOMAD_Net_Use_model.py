# -*- coding: utf-8 -*-
"""
Created on Mon May 11 19:33:51 2026

@author: Simone
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch, butter, filtfilt 
import scipy as sp
from scipy.signal import hilbert


#%%
# Load Data
# =============================================================================

df = pd.read_csv(r"Simulated_records.csv")
records = df.values
records = records.astype(float)

dens_neu = 64 # number of neurons per layer
ks = 31 # convolutional kernel layers
latent_dim = 16 # latent space dimension, mode shape


fs = 100 # sampling frequency
t = np.arange(0, len(records)/fs, 1/fs)


def build_hankel(data, n_lags):
    T, n_s = data.shape
    rows = T - n_lags + 1
    H = np.zeros((rows, n_s * n_lags))
    for lag in range(n_lags):
        H[:, lag*n_s:(lag+1)*n_s] = data[lag: lag + rows, :]
    return H


def get_hilbert_imag(real_part):
        N = real_part.shape[0]
        Xf = torch.fft.fft(real_part, n=N, dim=0)      
        h = torch.zeros(N, dtype=Xf.dtype, device=Xf.device)
        if N % 2 == 0:
            h[0] = 1
            h[N // 2] = 1
            h[1 : N // 2] = 2
        else:
            h[0] = 1
            h[1 : (N + 1) // 2] = 2
        h = h.view(-1, 1)
        analytic_signal = torch.fft.ifft(Xf * h, dim=0)
        return analytic_signal.imag

#%%
# Filter and resampling input if necessary
# =============================================================================

Fi = 0.1
Ft = 20
resampling = 2
order = 2


records = records - np.mean(records, axis=0)
nyq = fs / 2
ftype = 'bandpass'
b, a = butter(order, [Fi / nyq, Ft / nyq], btype=ftype)

for jj in range(records.shape[1]):
    records[:, jj] = filtfilt(b, a, records[:, jj])
records = records[::resampling, :]
fs = fs / resampling
t = np.arange(len(records)) / fs


#%%
# Plot input
# =============================================================================

# CPSD
nperseg = 1024
Nacc = records.shape[1]
Gyy = np.zeros((Nacc, Nacc, nperseg//2 + 1), dtype=complex)
for i in range(Nacc):
    for j in range(Nacc):
        f, Pxy = sp.signal.csd(records[:, i], records[:, j], fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        Gyy[i, j, :] = Pxy
sw = np.zeros(Gyy.shape[2])
uw = np.zeros((Gyy.shape[1], Gyy.shape[2]), dtype=complex)
for k in range(Gyy.shape[2]):
    U, S, Vh = np.linalg.svd(Gyy[:, :, k])
    sw[k] = S[0]         
    uw[:, k] = U[:, 0]    


# Plot original signal
plt.figure(figsize=(15, 8))
plt.subplot(3, 1, 1)
for n in range(np.size(records,1)):
    try:
        fname = f"Sensor_{n+1}.out"
        acc = records[:, n]
        plt.plot(t, acc, label=f"Sensor {n+1}")
    except Exception as e:
        print(f"Skipping sensor {n}: {e}")
plt.xlabel("t [s]")
plt.ylabel("a_x [g]")  
plt.title('Original signal')        
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend(loc="best", frameon=False)
plt.tight_layout()


# Plot Welch Spectrum
plt.subplot(3, 1, 2)
frequencies, power_spectrum = welch(records.T, fs=fs, nperseg=nperseg)
for n in range(np.size(records,1)):
    try:
        plt.semilogy(frequencies, power_spectrum[n,:].T, label=f"Sensor {n+1}")
    except Exception as e:
        print(f"Skipping sensor {n}: {e}")
plt.title('Welch Power Spectrum original signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power/Frequency [g²/Hz]')
plt.grid()
plt.legend(loc="best", frameon=False)
plt.tight_layout()


# Plot 1st singular value
plt.subplot(3, 1, 3)
plt.semilogy(f,sw)
plt.title('1st singular value')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power/Frequency [g²/Hz]')
plt.grid()
plt.tight_layout()
plt.show()


#%%
# Model
# =============================================================================

checkpoint = torch.load("Model.pt", map_location="cpu", weights_only=False)

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()


        pad = ks // 2
        

        self.act0 = nn.Tanh()

        # ENCODER
        self.enc1 = nn.Conv1d(NGDL_H, dens_neu, ks, padding=pad)
        self.enc2 = nn.Conv1d(dens_neu, dens_neu, ks, padding=pad)


        # BRANCH 1:
        self.fc_real = nn.Conv1d(dens_neu, N_MODES, kernel_size=1)


        # BRANCH 2:

        self.phi_conv1 = nn.Conv1d(dens_neu, dens_neu, ks, padding=pad)
        
        self.phi_conv2 = nn.Conv1d(dens_neu, dens_neu, ks, padding=pad)
        
        # Encoder
        self.phi_enc1 = nn.Linear(dens_neu, dens_neu)
        self.phi_enc2 = nn.Linear(dens_neu, latent_dim)
        
        # Decoder 
        self.phi_dec1 = nn.Linear(latent_dim, dens_neu)
        self.phi_dec2 = nn.Linear(dens_neu, NGDL * 2 * N_MODES)


    def forward(self, x):

        x = x.t().unsqueeze(0)

        # ENCODER COMUNE
        h = self.act0(self.enc1(x))
        h = self.act0(self.enc2(h))


        # BRANCH 1:
        y_real = self.fc_real(h)
        
        y_real = y_real - y_real.mean(dim=2, keepdim=True)
        y_real = y_real / (y_real.std(dim=2, keepdim=True) + 1e-8)
        
        y_real_tmp = y_real.squeeze(0).t()
        y_imag_tmp = get_hilbert_imag(y_real_tmp)
        y_imag = y_imag_tmp.t().unsqueeze(0)

        # BRANCH 2: AUTOENCODER FORME MODALI
        h_phi = self.act0(self.phi_conv1(h))
        h_phi = self.act0(self.phi_conv2(h_phi))      

        context = h_phi.mean(dim=2)                       
        
        # bottleneck 
        z_phi = self.act0(self.phi_enc1(context))       
        z_phi = self.phi_enc2(z_phi)                     
        
        # decoder 
        phi_dec = self.act0(self.phi_dec1(z_phi))        
        phi_dec = self.phi_dec2(phi_dec)                  
        
        phi_dec = phi_dec.view(NGDL, 2 * N_MODES)       
        
        phi_real = phi_dec[:, :N_MODES]                 
        phi_imag = phi_dec[:, N_MODES:]                
        
        norm_phi = torch.sqrt(
            torch.sum(phi_real**2 + phi_imag**2, dim=0, keepdim=True) + 1e-8
        )
        
        phi_real = phi_real / norm_phi
        phi_imag = phi_imag / norm_phi


        q_real = y_real.squeeze(0)   
        q_imag = y_imag.squeeze(0)   

        # RECONSTRUCTION
        outputreconstr = (
            phi_real @ q_real
            - phi_imag @ q_imag
        ).t()                                 

        Phi_complex = torch.complex(phi_real, phi_imag)

        return (
            y_real.squeeze(0).t(),         
            y_imag.squeeze(0).t(),         
            outputreconstr,      
            Phi_complex       
        )


mean_train = checkpoint["mean_train"]
std_train = checkpoint["std_train"]
N_LAGS = checkpoint["N_LAGS"]
N_MODES = checkpoint["N_MODES"]
NGDL = checkpoint["NGDL"]
NGDL_H = NGDL * N_LAGS
history = checkpoint['history']

model = MyNet()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

records_norm = (records - mean_train) / std_train

records_hankel = build_hankel(records_norm, N_LAGS)

input_hankel = torch.from_numpy(records_hankel).float()
t_hankel = t[:len(records_hankel)]

with torch.no_grad():
    output_real, output_imag, output_norm, Phi_complex = model(input_hankel)


test_preds_reconstruc = output_norm.cpu().numpy() * std_train + mean_train


ind_comp = output_real.cpu().numpy() + 1j * output_imag.cpu().numpy()

Phi_complex_np = Phi_complex.detach().cpu().numpy()

Phi_complex_np = std_train.ravel()[:, None] * Phi_complex_np
Phi_complex_np = Phi_complex_np / (np.sqrt(np.sum(np.abs(Phi_complex_np)**2, axis=0, keepdims=True)) + 1e-12)


#%% 
# Displaying a single sample for check reconstruction
# =============================================================================

for n in range(np.size(records,1)):
    sample_idx = n  
    plt.figure(figsize=(12,5))
    # Original
    plt.plot(t_hankel, records[:len(records_hankel), sample_idx], label='Original', alpha=0.7)
    # Reconstruction
    plt.plot(t_hankel, test_preds_reconstruc[:, sample_idx], label='Recostructed', alpha=0.7)
    plt.title(f"Comparison of Original vs. Reconstruction - Sensor {n+1}")
    plt.xlabel("Time [s]")
    plt.ylabel("acc [m/s2]")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    

#%%
# independent components
# =============================================================================

N_samples_tot = ind_comp.shape[0]
for n in range(N_MODES):
    signal = np.real(ind_comp[:, n])
    

    fft_values = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_values)
    
    freqs = np.fft.fftfreq(N_samples_tot, d=1/fs)
    half_idx = N_samples_tot // 2
    
    plt.figure(figsize=(12, 8))
    

    plt.subplot(2, 1, 1)
    plt.plot(t_hankel, signal, label=f'Independent Component {n+1}', alpha=0.7, color='blue')
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)

    plt.subplot(2, 1, 2)
    plt.plot(freqs[:half_idx], fft_magnitude[:half_idx], label=f"IC {n+1} (FFT)", color='red')
    plt.title('FFT - Frequency Spectrum')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.xlim([0, fs/2])
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(loc="best", frameon=False)
    
    plt.tight_layout()
    plt.show()


#%%
# MAC matrix
# =============================================================================

ris_ref = np.genfromtxt(r"OMA_CoV-SSI.csv", delimiter=',', dtype=str)
ris_ref = np.char.replace(ris_ref, 'i', 'j')
phi_ref = np.vectorize(complex)(ris_ref)[:,-NGDL:]


def mac_matrix(Phi_ref, Phi_test):
    """
    Parameter:
    - Phi_ref : np.ndarray, shape (n_dof, n_modes_ref)
    - Phi_test: np.ndarray, shape (n_dof, n_modes_test)
    
    - MAC: np.ndarray, shape (n_modes_ref, n_modes_test)
    """
    numerator = np.abs(np.dot(Phi_ref.T.conj(), Phi_test))**2  
    norm_ref = np.sum(np.abs(Phi_ref)**2, axis=0).reshape(-1, 1)
    norm_test = np.sum(np.abs(Phi_test)**2, axis=0).reshape(1, -1)  
    # MAC
    MAC = numerator / (norm_ref * norm_test)  
    return MAC


Phi_complex_np_ord = Phi_complex_np


mac_mat = mac_matrix(phi_ref.T, Phi_complex_np)
mac_mat_conj = mac_matrix(phi_ref.T, np.conj(Phi_complex_np))
mac_mat_conj1 = mac_matrix(np.conj(phi_ref.T), Phi_complex_np)

       
mac_stack = np.stack([mac_mat, mac_mat_conj, mac_mat_conj1], axis=0)
row_scores = mac_stack.max(axis=2)
best_version_per_row = np.argmax(row_scores, axis=0)     
n_rows = mac_mat.shape[0]
mac_mat_best = mac_stack[best_version_per_row, np.arange(n_rows), :]     


plt.figure(figsize=(15, 15))
cax = plt.imshow(mac_mat_best, cmap='gray_r', vmin=0, vmax=1)
for i in range(mac_mat_best.shape[0]):
    for j in range(mac_mat_best.shape[1]):
        plt.text(j, i, f"{mac_mat_best[i, j]:.2f}", ha='center', va='center', color='red', fontsize=10)

plt.tight_layout()
plt.xlabel('Network')
plt.ylabel('Classical-OMA')
plt.show()


#%%
# NExT - IDENTIFICATION
# =============================================================================

from scipy.fft import fft, ifft
from scipy.linalg import pinv, eig

def next_autocorr_1comp(s, dt, time_lag):


    s = np.asarray(s).squeeze() 
    s = np.real(s)
    s = s - np.mean(s)

    N = len(s) 
    M = int(round(time_lag / dt))

    nfft = 2 ** int(np.ceil(np.log2(2 * N - 1)))

    S = fft(s, nfft)
    R = ifft(S * np.conj(S))
    R = R[:M]

    norm = np.arange(N, N - len(R), -1)
    R = R / norm
    t = np.arange(len(R)) * dt

    return R, t


def build_hankel_itd(y, S):

    Y = np.asarray(y)  
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
 
    n_samples, n_channels = Y.shape  
    n_block_rows = 2 * S
    n_cols = n_samples - n_block_rows + 1

    H = np.zeros((n_block_rows * n_channels, n_cols),dtype=complex)    
    for i in range(n_block_rows):
        H[i*n_channels:(i+1)*n_channels, :] = Y[i:i+n_cols, :].T

    return H
    

def stableonly(V, D):

    idx = np.where(np.real(D) > 0)[0]
    V1 = V[:, idx]
    D1 = np.diag(D[idx])

    return V1, D1

def pol2mod(lam):

    lam = np.asarray(lam)
    if lam.ndim == 2:
        lam = np.diag(lam)

    lam = lam.ravel()
    M = len(lam)

    fz = np.full((2, M), np.nan, dtype=float)
    om0 = np.abs(lam)

    valid = np.isfinite(lam) & np.isfinite(om0) & (om0 > 0)

    fz[0, valid] = om0[valid] / (2 * np.pi)  
    fz[1, valid] = -np.real(lam[valid]) / om0[valid] 

    return fz


def ibrahim_itd_shift1(
    R,
    fs,
    f_peak=None,
    fmin=0.1,
    fmax=None,
    max_damping=0.10,
    force_return=True
    ):
    
    R = np.asarray(R).squeeze()
    R = np.reshape(R, (len(R), 1))
    
    Nr, K = R.T.shape
    
    S = 2
    S1 = S * Nr
    
    H = build_hankel_itd(R, S)
    
    H1 = H[:S1, :]
    H2 = H[Nr:S1 + Nr, :]   
    
    T11 = H1 @ H1.T
    T21 = H2 @ H1.T
    T12 = H1 @ H2.T
    T22 = H2 @ H2.T
    
    A1 = T21 @ pinv(T11)
    A2 = T22 @ pinv(T12)
    A = (A1 + A2) / 2
    
    mu, V = eig(A)
    
    dt = 1 / fs
    lam_c = np.log(mu) / dt
    
    fz = pol2mod(lam_c)
    
    fd_all = fz[0, :]
    zeta_all = fz[1, :]
    
    if fmax is None:
        fmax = fs / 2
    
    valid = (
        np.isfinite(fd_all)
        & np.isfinite(zeta_all)
        & (fd_all >= fmin)
        & (fd_all <= fmax)
        & (zeta_all > 0)
        & (zeta_all <= max_damping)
    )
    
    if np.any(valid):
    
        idx_valid = np.where(valid)[0]
    
        if f_peak is not None and np.isfinite(f_peak):
            j = idx_valid[np.argmin(np.abs(fd_all[idx_valid] - f_peak))]
        else:
            j = idx_valid[0]
    
        fd = float(fd_all[j])
        zeta = float(zeta_all[j])
    
        return fd, zeta
    
    if force_return:
    
        fallback = (
            np.isfinite(fd_all)
            & np.isfinite(zeta_all)
            & (fd_all >= fmin)
            & (fd_all <= fmax)
        )
    
        if np.any(fallback):
    
            idx_fallback = np.where(fallback)[0]
    
            if f_peak is not None and np.isfinite(f_peak):
                j = idx_fallback[np.argmin(np.abs(fd_all[idx_fallback] - f_peak))]
            else:
                j = idx_fallback[0]
    
            fd = float(fd_all[j])
            zeta = float(zeta_all[j])
    
            return fd, zeta
    
    return np.nan, np.nan


def welch_peak_1comp(x, fs, nperseg=1024, fmin=0.1, fmax=None):

    x = np.asarray(x).squeeze()
    x = np.real(x)
    x = x - np.mean(x)

    if fmax is None:
        fmax = fs / 2

    nperseg_eff = min(nperseg, len(x))
    if nperseg_eff < 8:
        return np.nan

    f, Pxx = welch(x, fs=fs, nperseg=nperseg_eff, return_onesided=True)

    idx = np.where((f >= fmin) & (f <= fmax))[0]

    if len(idx) == 0:
        return np.nan

    i_peak = idx[np.argmax(Pxx[idx])]

    return float(f[i_peak])

def extract_modal_params_from_IC(
    ind_comp,
    fs,
    time_lag=10,
    welch_nperseg=1024,
    fmin=0.1,
    fmax=None,
    max_damping=0.10,
    force_return = False
):

    ind_comp = np.asarray(ind_comp)

    if ind_comp.ndim == 1:
        ind_comp = ind_comp.reshape(-1, 1)

    dt = 1 / fs

    if fmax is None:
        fmax = fs / 2

    n_samples, n_ic = ind_comp.shape

    frqd = np.full(n_ic, np.nan, dtype=float)
    zetap = np.full(n_ic, np.nan, dtype=float)
    welch_peaks = np.full(n_ic, np.nan, dtype=float)

    for k in range(n_ic):

        x = ind_comp[:, k]
        x = np.real(x)
        x = x - np.mean(x)

        f_peak = welch_peak_1comp(
            x,
            fs=fs,
            nperseg=welch_nperseg,
            fmin=fmin,
            fmax=fmax
        )

        welch_peaks[k] = f_peak
        
        R, t = next_autocorr_1comp(x, dt=dt, time_lag=time_lag)


        fd, zeta = ibrahim_itd_shift1(
            R,
            fs=fs,
            f_peak=f_peak,
            fmin=fmin,
            fmax=fmax,
            max_damping=max_damping,
            force_return=force_return
        )

        if (not np.isfinite(fd)) or (not np.isfinite(zeta)):
            continue
        
        frqd[k] = float(fd)
        zetap[k] = 100 * float(zeta)

    idx_sort = np.argsort(np.where(np.isfinite(frqd), frqd, np.inf))

    frqd_sorted = frqd[idx_sort]
    zetap_sorted = zetap[idx_sort]
    welch_sorted = welch_peaks[idx_sort]
    ic_sorted = idx_sort + 1

    return frqd_sorted, zetap_sorted, welch_sorted, ic_sorted



time_lag = 10

if np.size(ind_comp,1) == NGDL:
    X = records[:ind_comp.shape[0], :Phi_complex_np.shape[0]]
    X_analytic = hilbert(X, axis=0)
    Q_projected = (pinv(Phi_complex_np)@ X_analytic.T).T
    
    frqd, zetap, welch_peaks, ic_order = (
        extract_modal_params_from_IC(
            ind_comp=Q_projected,
            fs=fs,
            time_lag=time_lag,
            welch_nperseg=1024,
            fmin=0.1,
            fmax=fs / 2,
            max_damping= 10 / 100,
            force_return=False
        )
    )
else:      
    frqd, zetap, welch_peaks, ic_order = extract_modal_params_from_IC(
        ind_comp=ind_comp,
        fs=fs,
        time_lag=time_lag,
        welch_nperseg=1024,
        fmin=0.1,
        fmax=fs / 2,
        max_damping= 10 / 100,
        force_return = False
    )

print("Frequencies [Hz]:", frqd)
print("Damping [%]:", zetap)

   
# PLOT 
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 5))

n_ic = ind_comp.shape[1]
cmap = plt.get_cmap("tab20")
colors = [cmap(i) for i in np.linspace(0, 1, n_ic)]

for k in range(n_ic):

    xk = np.real(ind_comp[:, k])
    xk = xk - np.mean(xk)

    f, Pxx = welch(xk,fs=fs,nperseg=1024,return_onesided=True)

    ax.plot(f,Pxx,color=colors[k],linewidth=1.2,label=f"IC {k+1}")

for f_id in frqd:
    if np.isfinite(f_id):
        ax.axvline(f_id, color="r", linestyle="--", linewidth=1)

for f_we in welch_peaks:
    if np.isfinite(f_we):
        ax.axvline(f_we, color="b", linestyle="--", linewidth=1)


ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Power/Frequency")
ax.grid(True)
ax.legend(loc="best", frameon=False, ncol=2)

fig.tight_layout()


