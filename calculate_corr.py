import numpy as np
from skimage import io
from tqdm import tqdm
import math
video_pt = io.imread('/scratch/sk10640/Dataset/PtCeO2.tif')

k = 5
num, H, W = video_pt.shape


corr_img = []
for img in tqdm(video_pt):
    img = img.astype('float64')
    avg_corr = 0.0
    num_ele = 0
    
    for i in range(k, H - k):
        for j in range(k, W - k):
            corr_pixels = set(img[i-k:i+k+1, j-k:j+k+1].flatten())
            corr_pixels_1 = set(img[i-k+1:i+k, j-k+1:j+k].flatten())
            pix_k = np.array(list(corr_pixels - corr_pixels_1))
            pix_ij =np.repeat(img[i][j], len(pix_k))
                        
            corr = np.dot(pix_ij, pix_k) /(np.dot(pix_ij, pix_ij) * np.dot(pix_k, pix_k))**0.5 
            
            if math.isnan(corr):
                continue
            
            num_ele += 1
            avg_corr +=  (corr - avg_corr)/num_ele
            
    
    corr_img.append(avg_corr)
    print(avg_corr)

                
print('Correlation for k = ', k, ' is ', np.mean(np.array(corr_img)))
