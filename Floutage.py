import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

def floutage(im, n = 20):
    """
    A partir d'un objet fichier image, renvoie un objet fichier image flouté
    n fois
    """
    M = np.asarray(im)

    # Copie de la matrice de départ
    M2 = np.array(M)

    for i in range(n):
        # Attention,on ne peut pas faire la somme, cela ne rentre plus dans un uint8 !
        M2[1:-1,1:-1,:] = 0.25*M2[2:,1:-1,:] + 0.25*M2[:-2,1:-1,:] + 0.25*M2[1:-1,:-2,:] + 0.25*M2[1:-1,2:,:]

    im2 = Image.fromarray(M2)

    return im2

if __name__ == "__main__":
    #Ouverture d'un fichier image sous forme d'un tableau
    im = Image.open('brad.png')

    im2 = floutage(im)

    #Affichage des deux images côte à côte
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(im)
    ax1.set_title("Image originale")

    ax2.imshow(im2)
    ax2.set_title("Image floutée")
    fig.savefig("brad_flou.png")
    plt.show()
