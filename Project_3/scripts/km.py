import pandas as pd
import numpy as np
import sys
sys.path.append(r'C:\2021-2022-Fall-semester\EECS522\Project')
from mlclass import mpp, knn, unsupervised_learning 




pic_pth = r'../flowersm.ppm'
from PIL import Image
im = Image.open(pic_pth)

np_im = np.array(im)


rows, cols = np_im.shape[0], np_im.shape[1]
image = np.reshape(np_im, (rows*cols,3))
                                          
mrows, ncols = image.shape

image = pd.DataFrame(image, columns = ['D1','D2','D3'])


k = 256
unsup = unsupervised_learning(image, k)

_, new_labels = unsup.Kmeans()





