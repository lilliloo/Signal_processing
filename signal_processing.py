# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 01:57:47 2020

@author: lilliloo
"""

# ----------------------------- #

#numtaps:
#pass_zero:
#N:データの個数
#dt:サンプリング周期
# ----------------------------- #

# Library
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Function
def fft(data, N, dt):
    F = np.fft.fft(data)
    freq = np.fft.fftfreq(N, d = dt)
    amp = np.abs(F/(N/2))
    return [F, freq, amp]

def lowpass(data, fc, dt):
    filter_low = signal.firwin(numtaps=21, cutoff = fc, fs = 1/dt)
    data_f = signal.lfilter(filter_low, 1, data)
    return data_f

def highpass(data, fc, dt):
    filter_high = signal.firwin(numtaps=51, cutoff = fc, fs = 1/dt, pass_zero = False)
    data_f = signal.lfilter(filter_high, 1, data)
    return data_f

def bandpass(data, fc_l, fc_h, dt):
    filter_band = signal.firwin(numtaps=51, cutoff=[fc_l, fc_h], fs = 1/dt, pass_zero = False)
    data_f = signal.lfilter(filter_band, 1, data)
    return data_f

def bandeliminate(data, fc_l, fc_h, dt):
    filter_band = signal.firwin(numtaps=31, cutoff=[fc_l, fc_h], fs = 1/dt)
    data_f = signal.lfilter(filter_band, 1, data)
    return data_f

def view_fft(data, N, dt):
    F, freq, amp = fft(data, N, dt)
    t = np.arange(0, N*dt, dt)
    #元の信号のプロット
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(2,1,1)
    ax.plot(t, x)
    ax.grid()
    ax.set_xlim(0,0.3)
    #元の信号の周波数解析の結果
    ax1= fig.add_subplot(2,1,2)
    ax1.plot(freq[1:int(N/2)], amp[1:int(N/2)])
    ax1.grid()
    fig.show()
    pass
    
def compare(original, freq_o, amp_o, filtered, freq_f, amp_f, N, dt):
    t = np.arange(0, N*dt, dt)
    #元の信号のプロット
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(2,1,1)
    ax.plot(t, x)
    ax.grid()
    ax.set_xlim(0,0.3)
    #元の信号の周波数解析の結果
    ax1= fig.add_subplot(2,1,2)
    ax1.plot(freq_o[1:int(N/2)], amp_o[1:int(N/2)])
    ax1.grid()
    #フィルターをかけた信号のプロット
    ax = fig.add_subplot(2,1,1)
    ax.plot(t, filtered)
    ax.grid()
    ax.set_xlim(0,0.3)
    #フィルター後の信号の周波数解析の結果
    ax1 = fig.add_subplot(2,1,2)
    ax1.plot(freq_f[1:int(N/2)], amp_f[1:int(N/2)])
    ax1.grid()
    ax1.set_xlim(0,350)    
    fig.show()
    pass

# Configuration parameter
N = 1024            # サンプル数
dt = 0.001          # サンプリング周期 [s]
f1, f2, f3 = 10, 60, 300 # 周波数 [Hz]

#サンプルデータ
t = np.arange(0, N*dt, dt) # 時間 [s]
x = 3*np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t) + 0.5*np.sin(2*np.pi*f3*t) # 信号

# Main
#元信号のプロットと周波数解析の結果
view_fft(x, N, dt)

#Lowpass
#FFT(元信号)
print("Low-pass filter")
F, freq, amp = fft(x, N, dt)
#信号に40Hzでローパスフィルターをかける
x_lpf = lowpass(x, 40, dt)
#FFT（フィルター後の信号）
F1, freq1, amp1 = fft(x_lpf, N, dt)
compare(x, freq, amp, x_lpf, freq1, amp1, N, dt)

#Highpass
print("High-pass filter")
#信号にハイパスフィルターをかける
x_hpf = highpass(x, 100, dt)
#FFT（フィルター後の信号）
F2, freq2, amp2 = fft(x_hpf, N, dt)
compare(x, freq, amp, x_hpf, freq2, amp2, N, dt)

#Bandpass
print("Band-pass filter")
#信号にバンドパスフィルターをかける
x_bpf = bandpass(x, 30, 100, dt)
#FFT（フィルター後の信号）
F3, freq3, amp3 = fft(x_bpf, N, dt)
compare(x, freq, amp, x_bpf, freq3, amp3, N, dt)

#Bandeliminate
print("Band-eliminate filter")
#信号にバンドエリミネートフィルターをかける
x_bef = bandeliminate(x, 30, 100, dt)
#FFT（フィルター後の信号）
F4, freq4, amp4 = fft(x_bef, N, dt)
compare(x, freq, amp, x_bef, freq4, amp4, N, dt)




































