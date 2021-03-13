import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from Conversion_niveaux_gris import niveaux_gris

def convolution_33(M, M_conv):
    """
    Applique à la matrice 2D M la matrice de convolution 3x3 M_conv.
    Renvoie une matrice 2D M2.
    """
    imax, jmax = M.shape

    M2 = np.array(M, dtype = np.int32)

    for i in range(1, imax-1):
        for j in range(1, jmax-1):
            somme = 0
            for n in range(3):
                for m in range(3):
                    somme += (M_conv[n, m] * M[i-(n-1), j-(m-1)])
            M2[i,j] = somme

    return M2

def convolution(M, M_conv):
    """
    Applique à la matrice 2D M la matrice de convolution M_conv.
    Renvoie une matrice 2D M2.
    La matrice de convolution doit être carrée de dimension impaire.
    """
    imax, jmax = M.shape

    M2 = np.array(M, dtype = np.int32)

    nmax = mmax = (M_conv.shape[0] - 1)//2
    nmin = mmin = -nmax
    print(nmax)

    for i in range(1, imax - nmax):
        for j in range(1, jmax - mmax):
            somme = 0
            for n in range(nmin, nmax + 1):
                for m in range(mmin, mmax + 1):
                    somme += (M_conv[n + nmin, m + mmin] * M[i-n, j-m])
            M2[i,j] = somme

    return M2

def sobel(im, seuil = 255):
    """
    A partir d'un objet fichier image, applique la méthode de Sobel de détection
    des contours. L'image doit être une image en niveaux de gris.
    """
    M = np.asarray(im) # matrice 3D RVB

    n, m, p = M.shape
    shape = n, m

    # Extraction d'une matrice 2D
    M2 = M[: ,: ,0]

    Gx = np.ndarray(shape, dtype = np.int32) # matrice des gradients suivant x
    Gy = np.ndarray(shape, dtype = np.int32) # matrice des gradients suivant y
    G = np.ndarray(shape, dtype = np.int32) # matrice des normes des gradients

    # matrices de convolution de l'algo de Sobel
    M_convx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    M_convy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # matrices de convolution de calcul du gradient
    #M_convx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    #M_convy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    Gx = convolution_33(M2, M_convx)
    Gy = convolution_33(M2, M_convy)

    G = np.abs(Gx) + np.abs(Gy) # Approximation de la norme

    return matrice_gradients_to_im(G)

def matrice_gradients_to_im(G):
    """
    A partir de la matrice des gradients, construit une image en niveaux de gris.
    """
    imax, jmax = G.shape
    M_sortie = np.ndarray((imax, jmax, 3), dtype = np.uint8)

    gmax = np.max(G) # on cherche le max de tous les gradients pour normalisation

    for i in range(imax):
        for j in range(jmax):
            M_sortie[i, j , 0] = M_sortie[i, j , 1] = M_sortie[i, j , 2] = G[i, j]*255/gmax

    im = Image.fromarray(M_sortie)

    return im

def conversion(im):
    """
    Convertit une image niveaux de gris à 1D en image niveaux de gris à 3D.
    """
    M = np.asarray(im)

    imax, jmax = M.shape

    shape = imax, jmax, 3

    M2 = np.ndarray(shape, dtype = np.uint8)
    M2[:, :, 0] = M2[:, :, 1] = M2[:, :, 2] = M[:, :]

    return Image.fromarray(M2)

def seuil(im, valeur):
    """
    A partir d'une image en niveaux de gris, renvoie l'image avec tous les
    points au dessus du seuil en blanc, les autres en noir.
    """
    M = np.asarray(im)
    imax, jmax, k = M.shape
    M2 = np.ndarray(M.shape, dtype = np.uint8)

    for i in range(imax):
        for j in range(jmax):
            if M[i, j, 0] > valeur:
                M2[i, j, 0] = M2[i, j, 1] = M2[i, j, 2] = 255
            else:
                M2[i, j, 0] = M2[i, j, 1] = M2[i, j, 2] = 0

    return Image.fromarray(M2)

def filtrage_gaussien(im, sigma):
    """
    Applique un filtrage gaussien 5x5 sur l'image.
    """
    # Construction de la matrice de convolution gaussienne
    imax = jmax = 5
    M_conv = np.ndarray((imax, jmax), dtype = np.float)
    for i in range(5):
        for j in range(5):
            M_conv[i, j] = np.exp( -((i-2)**2 + (j-2)**2)/(2*sigma**2) )

    M = np.asarray(im)
    M2 = convolution(M, M_conv)

    return Image.fromarray(M2)

if __name__ == "__main__":
    #Ouverture d'un fichier image sous forme d'un tableau
    im = Image.open('lena.png')

    #filtrage_gaussien(im, 0.625)

    #im2 = conversion(im)
    im2 = niveaux_gris(im)
    im3 = sobel(im2)
    Seuil = 20
    im4 = seuil(im3, Seuil)

    #Affichage des deux images côte à côte
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.imshow(im)
    ax1.set_title("Image originale")

    ax2.imshow(im2)
    ax2.set_title("Niveaux de gris")

    ax3.imshow(im3)
    ax3.set_title("Contours (Sobel)")

    ax4.imshow(im4)
    title = "Contours avec seuil = " + str(Seuil)
    ax4.set_title(title)

    plt.show(fig)
