import matplotlib.pyplot as plt
import numpy as np
from numpy import array

# sumrate vs SNR
fig1, ax1 = plt.subplots()
SNR = 10*np.log10(np.logspace(0.5,3,6))
allpass = [1.09603709, 2.1958357,  3.58552014, 5.16961044, 6.78666348, 8.34874453]
# svd = [ 2.11664617,  3.71246185,  5.42572665,  7.03303682,  8.73288853, 10.40114732]
svd = [1.21119306, 2.44896399, 3.91402446, 5.51660431, 7.12858825, 8.82250081]
upperbound = [ 2.22441074,  3.57474098,  5.1097665,   6.72557002,  8.37150949, 10.02764526]
ax1.plot(SNR, allpass, marker='o')
ax1.plot(SNR, svd, marker='o')
ax1.plot(SNR, upperbound, marker='o', linestyle='--')
ax1.grid(linestyle='--')
ax1.legend(['Allpass', 'SVD', 'Upper bound'])
ax1.set_xlabel('SNR [dB]')
ax1.set_ylabel('Sum Rate')



# sumrate vs number of users
fig2, ax2 = plt.subplots()
Nusers = [2,5,10,15,20]
svd = [array([6.31262816]), array([7.02278818]), array([7.11508697]), array([7.2646537]), array([7.21796027])] 
allpass = [array([4.28731627]), array([5.05794261]), array([5.49045601]), array([5.47350895]), array([5.42807268])]
ax2.plot(Nusers, allpass)
ax2.plot(Nusers, svd)
plt.show()