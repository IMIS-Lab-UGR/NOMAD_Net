
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 19:33:51 2026

@author: Simone
"""
import os
N_THREADS = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
import torch
torch.set_num_threads(N_THREADS)
torch.set_num_interop_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used:", device)
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch, butter, filtfilt 
import scipy as sp
import matplotlib.gridspec as gridspec
from pathlib import Path


#  Create a “results” folder if it does not exist
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)


#%%
# Net parameters
# =============================================================================

N_MODES = 4 # number of independent components
N_LAGS = 80 # Hankel lag number
shifts = np.arange(1, 120, 1) # SOBI shift diagonalisation
dens_neu = 64 # number of neurons per layer
ks = 31 # convolutional kernel layers
latent_dim = 16 # latent space dimension, mode shape
epoch = 100 # number of epoch
batch_size = 500 # batch size
lrs = 0.0001 # learning rate
wd = 1e-4 # weights decay
wm_s, wm_r, wm_rp, wm_phi = 1, 10, 1, 1  # Weight loss (MLD, MSE, RECURRENCE PLOT, MODE SHAPES)
dim = 2 # recurrence loss dimension
tou = 4 # recurrence loss time delay
pat = 20 # patience early stopping


#%%
# Load Data
# =============================================================================

df = pd.read_csv(r"simulated_records.csv")
records = df.values
records = records.astype(float)

fs = 100 # sampling frequency
t = np.arange(0, len(records)/fs, 1/fs)
NGDL = np.size(records,1) # number of degree of freedom


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

plt.savefig(results_dir / "input.png", dpi=300, bbox_inches="tight")
plt.close()

#%%
# Validation and training dataset
# =============================================================================

# split training and validation (20%)
val_size = int(len(records) * 0.2)
train_size = len(records) - val_size

train_data = records[:train_size, :]
val_data = records[train_size:, :]

NGDL_H = NGDL * N_LAGS

# Normalization
mean_train = np.mean(train_data, axis=0, keepdims=True)
std_train  = np.std(train_data,  axis=0, keepdims=True)

train_data_norm = (train_data - mean_train) / std_train
val_data_norm   = (val_data   - mean_train) / std_train

# Hankel
def build_hankel(data, n_lags):
    T, n_s = data.shape
    rows = T - n_lags + 1
    H = np.zeros((rows, n_s * n_lags))
    for lag in range(n_lags):
        H[:, lag*n_s:(lag+1)*n_s] = data[lag: lag + rows, :]
    return H

train_hankel = build_hankel(train_data_norm, N_LAGS)
val_hankel   = build_hankel(val_data_norm,   N_LAGS)  


train_tensor = torch.from_numpy(train_hankel).float()
val_tensor   = torch.from_numpy(val_hankel).float()
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=False, drop_last=True)
val_loader   = DataLoader(TensorDataset(val_tensor),   batch_size=batch_size, shuffle=False, drop_last=True)


#%%
# Define Net architecture
# =============================================================================

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

model = MyNet().to(device)


#%%
# Define loss function
# =============================================================================

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
    
def modal_shape_loss(Phi_complex):
   
    Phi_n = Phi_complex / (torch.sqrt(torch.sum(torch.abs(Phi_complex)**2, dim=0, keepdim=True)) + 1e-8)
    
    G = Phi_n.conj().T @ Phi_n
    I = torch.eye(G.shape[0],device=G.device,dtype=G.dtype)
    
    off_diag = G - I    
    loss_phi = torch.mean(torch.abs(off_diag)**2)

    return loss_phi
    
    
def rp_loss(y, dimension=2, time_delay=4, recurrence_rate=0.05):

    N, K = y.shape
    L = N - (dimension - 1) * time_delay
    emb = []
    for d in range(dimension):
        start = d * time_delay
        emb.append(y[start:start + L, :])

    S = torch.stack(emb, dim=2).permute(1, 0, 2)

    diff = S[:, :, None, :] - S[:, None, :, :]
    D2 = torch.sum(diff ** 2, dim=-1)

    with torch.no_grad():
        eps2 = torch.quantile(
            D2.reshape(K, -1),
            q=recurrence_rate,
            dim=1
        ).view(K, 1, 1).clamp_min(1e-8)


    R = torch.exp(-D2 / eps2)
    R0 = R[:, :-1, :-1]
    D_next = D2[:, 1:, 1:]

    num = torch.sum(R0 * D_next, dim=(1, 2))
    den = torch.sum(R0, dim=(1, 2)).clamp_min(1.0)

    loss_per_mode = num / den
    loss_per_mode = loss_per_mode / eps2.view(K)

    return loss_per_mode.mean()


def compute_loss(y_real, y_imag, x_rec, x_orig, Phi_complex, wm_s, wm_r, wm_rp, wm_phi, w_s, w_r, w_rp, w_phi, shifts=shifts):
    
    # MLD loss
    y_complex = torch.complex(y_real, y_imag)   
    N, n_ch = y_complex.shape
    sobi_loss = 0.0
    for tau in shifts:
        if tau >= N: continue
        R_tau = (y_complex[:-tau].t().conj() @ y_complex[tau:]) / (N - tau)
        off_diag = R_tau - torch.diag(torch.diag(R_tau))
        sobi_loss += torch.mean(torch.abs(off_diag)**2)

    # MSE
    mse_pure = torch.nn.functional.mse_loss(x_rec, x_orig[:, :NGDL])

    # Recurrence plot loss
    rp_loss_t = rp_loss(y_real,dimension=dim,time_delay=tou,recurrence_rate=0.05)
    
    # mode shape loss
    phi_loss_t = modal_shape_loss(Phi_complex)

    loss_s = wm_s * w_s * sobi_loss
    loss_r = wm_r * w_r * mse_pure
    loss_rp = wm_rp * w_rp * rp_loss_t
    loss_phi = wm_phi * w_phi * phi_loss_t

    return loss_s + loss_r + loss_rp + loss_phi, sobi_loss, mse_pure, rp_loss_t, phi_loss_t


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

    MAC = numerator / (norm_ref * norm_test)  
    return MAC

#%%
# EarlyStopping
# =============================================================================

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, path='best_model.h5'):
        self.patience = patience    
        self.min_delta = min_delta   
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

#%%
# Train model
# =============================================================================

ris_ref = np.genfromtxt(r"OMA_CoV-SSI.csv", delimiter=',', dtype=str)
ris_ref = np.char.replace(ris_ref, 'i', 'j')
phi_ref = np.vectorize(complex)(ris_ref)[:,[1+2,3+2,5+2,7+2]]

output_folder = "training_plots"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
if not os.path.exists(os.path.join(output_folder,"MAC")):
    os.makedirs(os.path.join(output_folder,"MAC"))


early_stopping = EarlyStopping(patience=pat, path='Model_Best_Weights.h5')

def train_model(train_loader, val_loader, model, epochs=epoch, warmup_epochs=5):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lrs, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    
    
    w_s, w_r, w_rp, w_phi = 1.0, 1.0, 1.0 ,1.0  # Dynamic weights
    # sma_s, sma_r, sma_o, sma_count = 0, 0, 0, 0
    # is_frozen = False

    history = {
        't_total': [], 'v_total': [],
        't_sobi': [], 't_rec': [], 't_rp': [], 't_phi': [],
        'v_sobi': [], 'v_rec': [], 'v_rp': [], 'v_phi': [],
        'epoch_axis': []
    }
    
    plt.ioff() 
    fig = plt.figure(figsize=(18, 11))

    for epoch in range(epochs):
        model.train()
        t_tot_ep, t_s_ep, t_r_ep, t_rp_ep, t_phi_ep = 0, 0, 0, 0, 0
        
        for batch in train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            yr, yi, xr, Phi_complex_t = model(inputs)
                       
            loss, s_pure, r_pure , rp_pure, phi_pure = compute_loss(yr, yi, xr, inputs, Phi_complex_t, wm_s, wm_r, wm_rp, wm_phi, w_s, w_r, w_rp, w_phi, shifts=shifts)
            
                    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            
            t_tot_ep += loss.item()
            t_s_ep += (wm_s * w_s * s_pure).item()
            t_r_ep +=  (wm_r * w_r * r_pure).item()
            t_rp_ep +=  (wm_rp * w_rp * rp_pure).item()
            t_phi_ep +=  (wm_phi * w_phi * phi_pure).item()


            # #  SMA: Accumulation during the warm-up
            # if not is_frozen:
            #     sma_s += s_pure.item(); 
            #     sma_r += r_pure.item(); 
            #     sma_rp += rp_pure.item();
            #     sma_phi += phi_pure.item();
            #     sma_count += 1

            # # FREEZE BASELINE: End of warmup
            # if not is_frozen and (epoch + 1) == warmup_epochs:
            #     w_s = 1.0 / ( (sma_s / sma_count) + 1e-8 )
            #     w_r = 1.0 / ( (sma_r / sma_count) + 1e-8 )
            #     w_rp = 1.0 / ( (sma_rp / sma_count) + 1e-8 )
            #     w_phi = 1.0 / ( (sma_phi / sma_count) + 1e-8 )
            #     is_frozen = True
            #     print(f"\n>>> SMA Frozen! W_SOBI: {w_s:.2e}, W_REC: {w_r:.2e}, W_RP: {w_rp:.2e}, W_PHI: {w_phi:.2e}")

              
        # validation
        model.eval()
        v_tot_ep, v_s_ep, v_r_ep, v_rp_ep, v_phi_ep = 0, 0, 0, 0, 0
        with torch.no_grad():
            for v_batch in val_loader:
                v_in = v_batch[0].to(device)
                vyr, vyi, vxr, Phi_complex_v = model(v_in)
                vl, vs, vr, vrp, vphi = compute_loss(vyr, vyi, vxr, v_in, Phi_complex_v,  wm_s, wm_r, wm_rp, wm_phi, w_s, w_r, w_rp, w_phi, shifts=shifts)
                
                v_tot_ep += vl.item()
                v_s_ep += (wm_s * w_s * vs).item()
                v_r_ep += (wm_r * w_r * vr).item()
                v_rp_ep += (wm_rp * w_rp * vrp).item()
                v_phi_ep += (wm_phi * w_phi * vphi).item()

                              
        # EARLY STOPPING
        avg_val_loss = v_tot_ep / len(val_loader)       
        early_stopping(avg_val_loss, model)       
        if early_stopping.early_stop:
            print(">>> Early stopping.")
            break    
        
         
        scheduler.step(avg_val_loss)
        
        
        # SAVE e PLOT 
        history['epoch_axis'].append(epoch)
        history['t_total'].append(t_tot_ep / len(train_loader))
        history['v_total'].append(v_tot_ep / len(val_loader))      
        history['t_sobi'].append(t_s_ep / len(train_loader))
        history['t_rec'].append(t_r_ep / len(train_loader))
        history['t_rp'].append(t_rp_ep / len(train_loader))
        history['t_phi'].append(t_phi_ep / len(train_loader))
        history['v_sobi'].append(v_s_ep  / len(val_loader))
        history['v_rec'].append(v_r_ep / len(val_loader))
        history['v_rp'].append(v_rp_ep / len(val_loader))
        history['v_phi'].append(v_phi_ep / len(val_loader))
            
        fig.clear()
        fig.suptitle(f'Training monitoring - Epoch {epoch}', fontsize=14)
        
             
        gs = gridspec.GridSpec(
            3, 4,
            figure=fig,
            height_ratios=[1.0, 1.4, 1.4],
            hspace=0.55,
            wspace=0.35
)

        # 1. Plot Loss
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(history['epoch_axis'], history['t_total'], label='Train')
        ax0.plot(history['epoch_axis'], history['v_total'], '--', label='Val')
        ax0.set_xlabel('Epoch')
        ax0.set_yscale('log')
        ax0.set_title('Total Loss')
        ax0.grid(True)
        ax0.legend()

        # 2. Loss training
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.plot(history['epoch_axis'], history['t_sobi'], label='SOBI-t', color='green')
        ax1.plot(history['epoch_axis'], history['t_rec'], label='MSE-t', color='red')
        ax1.plot(history['epoch_axis'], history['t_rp'], label='rp-t', color='blue')
        ax1.plot(history['epoch_axis'], history['t_phi'], label='phi-t', color='orange')
        ax1.set_xlabel('Epoch')
        ax1.set_yscale('log')
        ax1.set_title('Loss Components Train')
        ax1.grid(True)
        ax1.legend()
        
        # 3. Loss validation
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(history['epoch_axis'], history['v_sobi'],'--', label='SOBI-v', color='green')
        ax2.plot(history['epoch_axis'], history['v_rec'],'--', label='MSE-v', color='red')
        ax2.plot(history['epoch_axis'], history['v_rp'],'--', label='rp-v', color='blue')
        ax2.plot(history['epoch_axis'], history['v_phi'],'--', label='phi-v', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_yscale('log')
        ax2.set_title('Loss Components Val')
        ax2.grid(True)
        ax2.legend()

        # 4. Reconstruction last batch
        n_samples1 = v_in.shape[0]
        t_axis1 = np.arange(n_samples1) / fs
        ax3 = fig.add_subplot(gs[0, 3])
        ax3.plot(t_axis1,v_in[:batch_size, 0].cpu(), color='black', alpha=0.3, label='Original')
        ax3.plot(t_axis1,vxr[:batch_size, 0].detach().cpu(), '--', color='red', label='Reconstructed')
        ax3.set_xlabel('Sec')
        ax3.set_title('Signal Reconstruction (Sensor 1 - last val batch)')
        ax3.grid(True)
        ax3.legend()

        # 5. Indipendent component
        y_comp_plot = (vyr.cpu().numpy() + 1j * vyi.cpu().numpy())
        n_samples = y_comp_plot.shape[0]
        t_axis = np.arange(n_samples) / fs
        freqs = np.fft.fftfreq(n_samples, d=1/fs)
        half_idx = n_samples // 2
        

        ax_t = fig.add_subplot(gs[1, :])
        ax_t.set_title('Independent Components - Time histories')
        ax_t.set_xlabel('Sec')
        ax_t.set_ylabel('Amplitude')
        ax_t.grid(True)
        

        ax_f = fig.add_subplot(gs[2, :])
        ax_f.set_title('Independent Components - FFT')
        ax_f.set_xlabel('Hz')
        ax_f.set_ylabel('Magnitude')
        ax_f.grid(True)
        
        for n in range(N_MODES):
            signal = np.real(y_comp_plot[:, n])
            fft_mag = np.abs(np.fft.fft(signal))
        
            ax_t.plot(
                t_axis,
                signal,
                lw=0.8,
                label=f'IC {n+1}'
            )
        
            ax_f.plot(
                freqs[:half_idx],
                fft_mag[:half_idx],
                lw=0.8,
                label=f'IC {n+1}'
            )
        
        ax_t.legend(
            loc='upper right',
            fontsize=7,
            frameon=False,
            ncol=2
)
        
        fig.savefig(
          os.path.join(output_folder, f"training_epoch_{epoch:04d}.png"),
          dpi=300,
          bbox_inches="tight"
      )
        
                
        Phi_complex_np_train = Phi_complex_v.detach().cpu().numpy()     

        Phi_complex_np_train = std_train.ravel()[:, None] * Phi_complex_np_train
        Phi_complex_np_train = Phi_complex_np_train / (np.sqrt(np.sum(np.abs(Phi_complex_np_train)**2, axis=0, keepdims=True)) + 1e-12)

        mac_mat = mac_matrix(phi_ref.T, Phi_complex_np_train)
        mac_mat_conj = mac_matrix(phi_ref.T, np.conj(Phi_complex_np_train))
        mac_mat_conj1 = mac_matrix(np.conj(phi_ref.T), Phi_complex_np_train)
        
               
        mac_stack = np.stack([mac_mat, mac_mat_conj, mac_mat_conj1], axis=0) 
        row_scores = mac_stack.max(axis=2)        
        best_version_per_row = np.argmax(row_scores, axis=0)      
        n_rows = mac_mat.shape[0]
        mac_mat_best = mac_stack[best_version_per_row, np.arange(n_rows), :]     
       
              
        plt.figure(figsize=(15, 15))    
        for i in range(mac_mat_best.shape[0]):
            for j in range(mac_mat_best.shape[1]):
                plt.text(j, i,f"{mac_mat_best[i, j]:.2f}",ha='center',va='center',color='red',fontsize=10)
        
        plt.tight_layout()
        plt.xlabel('Network')
        plt.ylabel('OMA-FEM')

        plt.savefig(os.path.join(output_folder,"MAC", f"MAC_{epoch:04d}.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
        print(
                f"Epoch {epoch} | "
                f"Total Train: {history['t_total'][-1]:.6f} | "
                f"Total Val: {history['v_total'][-1]:.6f} | "
                f"SOBI_t: {history['t_sobi'][-1]:.6f} | "
                f"MSE_t: {history['t_rec'][-1]:.6f} | "
                f"RP_t: {history['t_rp'][-1]:.6f} | "
                f"PHI_t: {history['t_phi'][-1]:.6f} | "
)
    return history

# Run training
history = train_model(train_loader, val_loader, model)
model.load_state_dict(torch.load(early_stopping.path))


#%%
# Save or load model
# =============================================================================

# Save
checkpoint = {
    "model_state_dict": model.state_dict(),
    "history": history,
    "mean_train": mean_train,
    "std_train": std_train,
    "N_LAGS": N_LAGS,
    "N_MODES": N_MODES,
    "NGDL": NGDL,
    "fs": fs,
    "Fi": Fi,
    "Ft": Ft,
    "resampling": resampling,
    "order": order
}
torch.save(checkpoint, 'Model.pt')


# # Load
# model = MyNet()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint = torch.load("Model.pt",map_location=device,weights_only=False)
# model.load_state_dict(checkpoint['model_state_dict'])
# history = checkpoint['history']


#%%
# Evaluate model
# =============================================================================

eval_batch_size = int(fs * 1800)

records_hankel = build_hankel((records[:eval_batch_size,:] - mean_train) / std_train, N_LAGS)
input_hankel = torch.from_numpy(records_hankel).float().to(device)

with torch.no_grad():
    output_real, output_imag, output_norm, Phi_complex = model(input_hankel)
    
    t_hankel = t[:len(records_hankel)]
    test_preds_reconstruc = (output_norm.cpu().numpy() * std_train) + mean_train


#%% plot
# Displaying a single sample for check reconstruction
# ===============================

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
    
    plt.savefig(results_dir / f"reconstruction_sensor_{n}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
#%%
# Extract mode shapes
# =============================================================================

Phi_complex_np = Phi_complex.detach().cpu().numpy()
Phi_complex_np = std_train.ravel()[:, None] * Phi_complex_np
Phi_complex_np = Phi_complex_np / (np.sqrt(np.sum(np.abs(Phi_complex_np)**2, axis=0, keepdims=True)) + 1e-12)
    
#%%
# Extract independent components
# =============================================================================

ind_comp = (output_real.detach().cpu().numpy() + 1j * output_imag.detach().cpu().numpy())
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
    
    plt.savefig(results_dir / f"ind_component_{n}.png", dpi=300, bbox_inches="tight")
    plt.close()

#%%
# ===============================
# MAC matrix
# ===============================

ris_ref = np.genfromtxt(r"OMA_SSI.csv", delimiter=',', dtype=str)
ris_ref = np.char.replace(ris_ref, 'i', 'j')
phi_ref = np.vectorize(complex)(ris_ref)[:,-NGDL:]


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

plt.savefig(results_dir / "MAC.png", dpi=300, bbox_inches="tight")
plt.close()

      
