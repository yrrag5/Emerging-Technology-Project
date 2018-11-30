# Imports Used 
import numpy as np
import matplot.lib.pylot as plt
import pandas as pd
#import keras

import gzip

with gzip.open('Data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    imgContent = f.read()
	
with gzip.open('Data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    labelContent = f.read()	
	

with gzip.open('Data/train-images-idx3-ubyte.gz', 'rb') as f:
    tImage = f.read()
	

with gzip.open('Data/train-labels-idx3-ubyte', 'rb') as f:
    tLabel = f.read()
	
tImage = ~np.array(list(imgContent[16:])).reshap(60000, 28 ,28).astype(np,uint8) / 255.0

tLabel = np.array(list(labelContent[8:])).astype(np.uint8)
	



