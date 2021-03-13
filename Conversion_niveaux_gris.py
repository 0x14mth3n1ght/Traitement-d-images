import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image


def niveaux_gris(im):
    """
    Renvoie un objet File image qui correspond à l'objet File image passé en
    argument converti en niveaux de gris
    Ici l'image finale pèse aussi lourd que l'image initiale
    """
    # matrice de l'image couleur
    M = np.asarray(im)

    colonnes, lignes = im.size

    # Création d'un tableau de float rempli de zéros
    M2 = np.ndarray((lignes,colonnes,3), dtype = np.uint8)

    for i in range(lignes):
        for j in range(colonnes):
            # Attention,on ne peut pas faire la somme sans coefficients,
            #cela ne rentre plus dans un uint8 !
            # Ici, les coeff sont ceux qui donnent une bonne luminance
            M2[i,j,0] = M2[i,j,1] = M2[i,j,2] = 0.2126*M[i,j,0] + 0.7152*M[i,j,1] + 0.0722*M[i,j,2]

    # Création de l'objet File à partir du tableau
    im2 = Image.fromarray(M2)

    return im2

if __name__ == "__main__":
    im = Image.open('lena.png')
    im2 = niveaux_gris(im)

    #Affichage des deux images côte à côte
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(im)
    ax2.imshow(im2)
    plt.show(fig)
