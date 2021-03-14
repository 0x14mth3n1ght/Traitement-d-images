# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

def test():
    M = np.ndarray((256,256,3), dtype = np.uint8)

    # Dégradé
    for i in range(256):
        for j in range(256):
            M[i,j,0] = i
            M[i,j,1] = j
            M[i,j,2] = 255
            
            
    t = np.ndarray((256,256,3), dtype = np.uint8)
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

    im = Image.fromarray(M)
    im2= Image.fromarray(t)
    
    fig,ax = plt.subplots(1,2)
    
    ax[0].imshow(im)
    ax[1].imshow(im2)
    plt.show(fig)

def niveaux_gris(im):
    M = np.asarray(im)
    M1 = np.zeros(M.shape, dtype = np.uint8)
    n, m, k = M.shape
    for i in range(n):
        for j in range(m):
            R,V,B = M[i,j]  #-M[i;j] pour un negatif, 4eme canal pour certaines images
            M1[i,j] = R//3 + V//3 + B//3
    
    im2 = Image.fromarray(M1)
    return im2

def floutage(im ,n):
    M = np.asarray(im)
    M1 = np.array(M)
    for i in range(n): #integration = filtrage passe-bas
        M1[1:-1,1:-1,:] = 0.25*M1[2:,1:-1,:] + 0.25*M1[:-2,1:-1,:] + 0.25*M1[1:-1,:-2,:] + 0.25*M1[1:-1,2:,:] 
    im2 = Image.fromarray(M1)
    return im2

def contours(im):
    M = np.asarray(im)
    N=niveaux_gris(M)
    Mc = np.array(N, dtype=np.int32)
    M1 = np.zeros(M.shape, dtype=np.int32)
    M2 = np.zeros(M.shape, dtype=np.int32)
    M1[1:-1, :, :] = Mc[2:,:,:] - Mc[:-2,:,:] #derivation = filtrage passe-haut
    M2[:, 1:-1, :] = Mc[:,2:,:] - Mc[:,:-2,:]
    M3 = np.abs(M1) + np.abs(M2)    #norme du grad de l'intensite lumineuse
    M4 = np.array(M3, dtype=np.uint8)
    im4 = Image.fromarray(M4)
    return M4

def tranche(xA, yA, xB, yB, M):
    """
    La fonction prend en argument deux points A et B et un tableau représentant
    une image couleur convertie en niveaux de gris.
    La fonction trace sur l'image modifiable passée en argument la droite
    passant par A et B en négatif.
    La fonction renvoie l'image modifiée, et deux listes X et Z qui permettent
    ensuite de tracer Z = f(X) (affichage du profil de la tranche).
    """
    #On créé une copie modifiable de ce tableau (l'original ne l'est pas) pour
    #pouvoir dessiner la droite
    M_copie = np.array(M, dtype = np.uint8)

    X = []
    Z = []

    xmax, ymax = M_copie.shape[0:2]

    delta_x = abs(xB - xA)
    delta_y = abs(yB - yA)

    if (delta_x >= delta_y):
        a = (yB - yA)/(xB - xA)
        b = yA - a*xA
        for x in range(xmax):
                y = round(a*x + b)
                if (y >= 0) and (y <= ymax):
                    X.append(x)
                    Z.append(M_copie[x,y,0])
                    M_copie[x,y,:] = 255 - M_copie[x,y,:]

    else:
        a = (xB - xA)/(yB - yA)
        b = xA - a*yA
        for y in range(ymax):
                x = round(a*y + b)
                if (x >= 0) and (x <= xmax):
                    X.append(y)
                    Z.append(M_copie[x,y,0])
                    M_copie[x,y,:] = 255 - M_copie[x,y,:]

    return M_copie, X, Z

im = Image.open('lena.png')

fig, ax = plt.subplots(1, 5, figsize=(10, 30))   #fenetre d'affichage et zone de trace
ax[0].imshow(im)          #creation du graphique a partir de l'objet fichier

im1 = niveaux_gris(im)
ax[1].imshow(im1)

im2 = floutage(im, 30)
ax[2].imshow(im2)
    
im3 = contours(im)
ax[3].imshow(im3)
    
im4=tranche(20, 30, 10, 20, np.asarray(im))
ax[4].imshow(im4)