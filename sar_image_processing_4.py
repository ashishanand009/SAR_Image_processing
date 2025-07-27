from scipy.fft import fft2, fftshift
import scipy
from skimage.color import rgb2gray
from skimage.filters import window
from skimage.color import rgb2gray
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sarpy.io.complex.converter import open_complex

sicd_path = Path("D:\\SAR_dataset\\IMG-VV-STRIXB-20220811T004713Z-SMSLC-SICD.nitf")
reader = open_complex(str(sicd_path))
print('image size as tuple ={}'.format(reader.get_data_size_as_tuple())) 
slc = reader[:] 
print(slc.dtype, slc.shape) 
x=np.zeros(36699);
slc_fdom = np.zeros(slc.shape,dtype=np.complex_);
fdom_imag = np.zeros(slc.shape,dtype=np.complex_);
ifft_fdom_imag = np.zeros(slc.shape,dtype=np.complex_); 

#displaying image with mean to show contrast
A=np.abs(slc)
Am=np.mean(A)
As=np.std(np.abs(A))
plt.figure(figsize=(4,4))
plt.imshow(np.abs(A),cmap='gray',vmin=0,vmax=Am+As)
plt.savefig("SAR_image_before.jpg");
#starting windowing 
for i in range(0,4388):
    x= slc[i,:];
    slc_fdom[i,:]=np.fft.fft(x); # fft per row of image pixel
    fdom_imag[i,:]=slc_fdom[i,:]; 
for i in range (0,4388):
    #doing frequency domain windowing around center of frequency
    fdom_imag[i,18000:22000]=slc_fdom[i,18000:22000] * window('hann', 4000);
for i in range(0,4388):
    #taking ifft of widowed freq domain of image 
    ifft_fdom_imag[i,:]=np.fft.ifft(fdom_imag[i,:]);

print(slc_fdom.dtype, slc_fdom.shape) 

#displaying image with mean to show contrast
A_ifft=np.abs(ifft_fdom_imag)
Am_ifft=np.mean(A_ifft)
As_ifft=np.std(np.abs(A_ifft))
plt.figure(figsize=(4,4))
plt.imshow(np.abs(A_ifft),cmap='gray',vmin=0,vmax=Am_ifft+As_ifft)
plt.savefig("SAR_image_after.jpg");