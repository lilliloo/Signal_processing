# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 23:26:52 2020

@author: lilliloo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def view_all_order(t, sig, sig_noise, order_list):
    axes_count = 4 # グラフの個数   
    fig = plt.figure(figsize=(12,8))   
    for i in range(axes_count):
        # ピーク検出
        maximal_idx = signal.argrelmax(sig_noise, order=order_list[i]) # 極大値インデックス取得
    
        fig.add_subplot(2, 2, i+1)
        plt.title('order={}'.format(order_list[i]), fontsize=18)
        plt.plot(t, sig,'r',label='iriginal', alpha=0.5) # 正弦波
        plt.plot(t, sig_noise,label='original + noise') # 正弦波＋ノイズ
    
        plt.plot(t[maximal_idx],sig_noise[maximal_idx],'ro',label='peak_maximal') # 極大点プロット   
    plt.tight_layout()
    pass
np.random.seed(0) # 乱数seed固定

N = 512 # サンプル数
dt = 0.01 # サンプリング間隔
fq1, fq2 = 3, 8 # 周波数
amp1, amp2 = 1.5, 1 # 振幅 

t = np.arange(0, N*dt, dt) # 時間軸
# 時間信号作成
sig = amp1*np.sin(2*np.pi*fq1*t) + amp2*np.sin(2*np.pi*fq2*t)
sig_noise = sig + np.random.randn(N)*0.5
freq = np.linspace(0, 1.0/dt, N) # 周波数軸

# 高速フーリエ変換
F = np.fft.fft(sig)

F_abs = np.abs(F) # 複素数 ->絶対値に変換
# 振幅を元の信号のスケールに揃える
F_abs = F_abs / (N/2) # 交流成分
F_abs[0] = F_abs[0] / 2 # 直流成分
    
# グラフ表示（時間軸）
plt.figure(1, figsize=(15,10))

plt.subplot(211)
plt.plot(t, sig, label = "original", alpha = 0.5)
plt.plot(t, sig_noise, label = "original + noise")
plt.xlabel('Time(s)')
plt.ylabel('Signal')

maximal_idx = signal.argrelmax(sig_noise, order=4) # 極大値インデックス取得
minimal_idx = signal.argrelmin(sig_noise, order=4) # 極小値インデックス取得

plt.plot(t[maximal_idx],sig_noise[maximal_idx],'ro',label='peak_maximal') # 極大点プロット
plt.plot(t[minimal_idx],sig_noise[minimal_idx],'bo',label='peak_minimal') # 極小点プロット
plt.legend(loc ="best")
# 極大値情報の表示
print("----maximal-----")
print('idx_length:', len(maximal_idx)) # peakの検出数
print('idx_value:', maximal_idx) # peakのindex
print('x_value:', t[maximal_idx]) # peakのx値
print('y_value:', sig_noise[minimal_idx]) #peakのy値

print("----minimal-----")
print('idx_length:', len(minimal_idx)) # peakの検出数
print('idx_value:', minimal_idx) # peakのindex
print('x_value:', t[minimal_idx]) # peakのx値
print('y_value:', sig_noise[minimal_idx]) #peakのy値


## FFTデータからピークを自動検出
maximal_idx = signal.argrelmax(F_abs, order=1)[0] # ピーク（極大値）のインデックス取得
# ピーク検出感度調整、後半側（ナイキスト超）と閾値より小さい振幅ピークを除外
peak_cut = 0.3 # ピーク閾値
maximal_idx = maximal_idx[(F_abs[maximal_idx] > peak_cut) & (maximal_idx <= N/2)]
plt.subplot(212)
## グラフ表示（周波数軸）
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.axis([0,1.0/dt/2,0,max(F_abs)*1.5])
plt.plot(freq, F_abs)
plt.plot(freq[maximal_idx], F_abs[maximal_idx],'ro',label = "peak")
plt.xlim([0,30])
plt.legend()

# グラフにピークの周波数をテキストで表示
for i in range(len(maximal_idx)):
    plt.annotate('{0:.0f}(Hz)'.format(np.round(freq[maximal_idx[i]])),
                 xy=(freq[maximal_idx[i]], F_abs[maximal_idx[i]]),
                 xytext=(10, 20),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2")
                )
#
plt.subplots_adjust(hspace=0.4)
plt.show()
print('peak', freq[maximal_idx])

# orderごとのpeak
order_list = [1, 2, 4, 8]
view_all_order(t, sig, sig_noise, order_list)
