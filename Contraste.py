import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

def hist_img(M):
    h = np.zeros((3, 256), dtype = np.int32)
    n, m, k = M.shape
    for i in range(n):
        for j in range(m):
            h[0,M[i,j,0]]+=1
            h[1,M[i,j,1]]+=1
            h[2,M[i,j,2]]+=1
    return h

def plot_hist_img(M, fichier=None): 
    """ Affiche les trois histogrammes RVB de l'image M. Sauve l'image dans le fichier spécifié. """ 
    #h_R, h_V, h_B = hist_img(M) 
    h = hist_img(M) 
    X = np.linspace(0, 255, 256) 
    colors = ('red', 'green', 'blue')
    
    fig, ax = plt.subplots(1, 3, figsize = (30, 10)) 
    for c in range(3): 
        ax[c].plot(X, h[c], color = colors[c], alpha = 0.7)
    
def seuils(h, percent=0.01): 
     """ Renvoie les seuils a et b pour augmentation du contraste. h est un vecteur histogramme (un vecteur d'entiers de taille 256). percent est le pourcentage de pixels à parcourir pour déterminer a et b. """ 
     nb_pix = np.sum(h)
     
     nb_points = 0 
     a = 0 
     while nb_points < nb_pix*percent: 
         nb_points += h[a] 
         a += 1
    
     nb_points = 0
     b = len(h)-1
     while nb_points < nb_pix*percent: 
         nb_points += h[b] 
         b -= 1
     
     return a, b

def aug_contraste(M, percent = 0.01): 
     """ Renvoie la matrice image après augmentation du contraste. La dynamique finale est maximale, elle appartient à [0, 255] """ 
     Mc = M.copy()
     Mc = Mc.astype(np.float64)
    
     #h_R, h_V, h_B = hist_img(M) 
     h = hist_img(M)
     
     a, b = np.zeros((3), np.int32), np.zeros((3), np.int32) 
     for c in range(3): 
         a[c], b[c] = seuils(h[c], percent) 
         #a_R, b_R = seuils(h_R, percent) 
         #a_V, b_V = seuils(h_V, percent) 
         #a_B, b_B = seuils(h_B, percent)
         
         #R, V, B = Mc[:,:,0], Mc[:,:,1], Mc[:,:,2]
     
     for c in range(3): 
            Mc[:,:,c] = np.rint( 255 * ((Mc[:,:,c] - a[c]) / (b[c] - a[c])) ) 
            Mc[:,:,c][Mc[:,:,c] < 0] = 0
            Mc[:,:,c][Mc[:,:,c] > 255] = 255
            #Mc[:,:,0] = np.rint( 255 * ((R - a_R) / (b_R - a_R)) ) 
            #Mc[:,:,1] = np.rint( 255 * ((V - a_V) / (b_V - a_V)) ) 
            #Mc[:,:,2] = np.rint( 255 * ((B - a_B) / (b_B - a_B)) )

            # Saturation des valeurs négatives à 0 
            #Mc[:,:,0][Mc[:,:,0] < 0] = 0 
            #Mc[:,:,1][Mc[:,:,1] < 0] = 0 
            #Mc[:,:,2][Mc[:,:,2] < 0] = 0
            
            # Saturation des valeurs négatives à 255 
            #Mc[:,:,0][Mc[:,:,0] > 255] = 255 
            #Mc[:,:,1][Mc[:,:,1] > 255] = 255 
            #Mc[:,:,2][Mc[:,:,2] > 255] = 255
        
     Mc = Mc.astype(np.uint8)
     return Mc
 
im = Image.open('ecrevisse.png') 
#im = Image.open('./Images/lena.png')

M = np.asarray(im) 
plot_hist_img(M, 'hist_original.png')

# Essais avec différents pourcentages 
# On remarque que les valeurs 0 et 255 sont de plus en plus peuplées 
# les trous entre valeurs possibles augmentent en taille 
Mc = aug_contraste(M, percent = 0.001) 
Mc2 = aug_contraste(M, percent = 0.01) 
Mc3 = aug_contraste(M, percent = 0.05) 

plot_hist_img(Mc)
plot_hist_img(Mc2, 'hist_modifié.png')
plot_hist_img(Mc3)

im = Image.fromarray(M) 
imc = Image.fromarray(Mc) 
imc2 = Image.fromarray(Mc2) 
imc3 = Image.fromarray(Mc3) 

fig, ax = plt.subplots(2, 2, figsize = (20, 15)) 
ax[0, 0].imshow(im) 
ax[0, 1].imshow(imc) 
ax[1, 0].imshow(imc2) 
ax[1, 1].imshow(imc3)
ax[0, 0].set_title('Image originale') 
ax[0, 1].set_title('0.1%') 
ax[1, 0].set_title('1%') 
ax[1, 1].set_title('5%')

fig, ax = plt.subplots() 
ax.imshow(imc3) 
plt.savefig('panneau-brouillard_aug.jpg')
