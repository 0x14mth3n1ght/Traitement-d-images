import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

M = np.ndarray((256,256,3), dtype = np.uint8)

# Dégradé
for i in range(256):
    for j in range(256):
        M[i,j,0] = i
        M[i,j,1] = j
        M[i,j,2] = 255
"""
# Cercle de rayon R et de centre (xc,yc)
R = 50
xc = int(256/2)
yc = int(256/2)

for i in range(256):
    for j in range(256):
        if (i-xc)**2 + (j-yc)**2 <= R**2:
            t[i,j,0] = 255
            t[i,j,1] = 0
            t[i,j,2] = 0
        else:
            t[i,j,0] = 0
            t[i,j,1] = 0
            t[i,j,2] = 255
"""
im = Image.fromarray(M)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(im)
plt.show(fig)
