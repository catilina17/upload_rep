import numpy as np
import time
from skimage.util.shape import view_as_windows as viewW

lg=120
n=60
ht=1000

mat=np.arange(1,lg+1).reshape(1,lg)

mat=np.concatenate([np.concatenate([mat]*n,axis=0).reshape(1,n,lg)]*ht,axis=0)
timo = time.time()
mat=np.swapaxes(mat,1,2)
mat=np.swapaxes(mat,1,2)

print(time.time()-timo)
import sys
sys.exit(0)
mat2=mat.copy()
shift=[i for i in range(0,n)]



def shift_by(data,shifto,axis):
    _max=data.shape[axis]
    for i in range(1, _max):
        _shift=shifto[i]
        data[:, i, :] = np.roll(data[:, i, :], _shift, axis=1)
        data[:, i, :_shift] = 0
timo = time.time()
shift_by(mat,shift,1)
print(time.time()-timo)

timo = time.time()

for i in range(1, n):
    """ 2.2 Pour chaque PN émise au mois i, on décale le profil d'écoulement de i """
    mat2[:, i, :] = np.roll(mat2[:, i, :], i, axis=1)
    mat2[:, i, :i] = 0

print(time.time()-timo)

print((mat == mat2).all())





