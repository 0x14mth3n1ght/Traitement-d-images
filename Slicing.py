import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from Conversion_niveaux_gris import niveaux_gris

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

def affiche_profil_tranche(im, xA, yA, xB, yB):
    """
    La fonction affiche l'image "tranchée" et le profil de la tranche
    à partir de l'objet fichier im
    """
    M = np.asarray(im, dtype = np.uint8)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    #Affichage de la tranche
    M_copie, X, Z = tranche(xA, yA, xB, yB, M)
    im2 = Image.fromarray(M_copie) # image tranchée

    #Affichage de l'image
    ax1.imshow(im2)

    ax2.plot(X, Z, color = "blue", alpha = 0.7, linewidth = 1.0, linestyle="-")
    ax2.set_xlim(0,len(X))
    ax2.set_ylim(0,255)

    fig.show()

    # On sauve l'image obtenue
    #fig.savefig("slicing.png")


if __name__ == "__main__":
    #On ouvre un fichier image
    im = Image.open('diffraction.png')

    # On le convertit en niveaux de gris pour ne pas avoir trois tranches (RVB)
    # mais une seule
    im2 = niveaux_gris(im)

    # Coordonnées des deux points qui définissent la tranche, attention inversion
    # des axes x et de y
    xA = 78
    yA = 30
    xB = 78
    yB = 200

    affiche_profil_tranche(im2, xA, yA, xB, yB)
