import matplotlib.pyplot as plt

def appartient_au_triangle(X, Y, x, y):
    """verifie si le point P(x;y) appartient au triangle ABC"""
   
    C=(Y[1]-Y[0])*X[2]+(X[0]-X[1])*Y[2]-((Y[1]-Y[0])*X[0]+(X[0]-X[1])*Y[0])
    c=(Y[1]-Y[0])*x+(X[0]-X[1])*y-((Y[1]-Y[0])*X[0]+(X[0]-X[1])*Y[0])

    A=(Y[2]-Y[1])*X[0]+(X[1]-X[2])*Y[0]-((Y[2]-Y[1])*X[1]+(X[1]-X[2])*Y[1])  
    a=(Y[2]-Y[1])*x+(X[1]-X[2])*y-((Y[2]-Y[1])*X[1]+(X[1]-X[2])*Y[1])

    B=(Y[0]-Y[2])*X[1]+(X[2]-X[0])*Y[1]-((Y[0]-Y[2])*X[2]+(X[2]-X[0])*Y[2])
    b=(Y[0]-Y[2])*x+(X[2]-X[0])*y-((Y[0]-Y[2])*X[2]+(X[2]-X[0])*Y[2])

    if C*c>=0 and B*b>=0 and A*a>=0:
        return True
    else:
        return False
   
def test(n, xmin, xmax, ymin, ymax, X, Y):
    hx=(xmax-xmin)/n
    hy=(ymax-ymin)/n
    
    X1=[]
    Y1=[]
    X2=[]
    Y2=[]
    
    for i in range(n):
        for j in range(n):
            x = xmin + hx*i
            y = ymin + hy*j
            
            if appartient_au_triangle(X, Y, x, y):
                X1.append(x)
                Y1.append(y)
            else:
                X2.append(x)
                Y2.append(y)
    
    fig,ax=plt.subplots()
    ax.scatter(X1,Y1,color='green',s=3)
    ax.scatter(X2,Y2,color='red',s=3)

print(test(50,0.,10,0.,10,[1,5,7],[7,1,5]))

    
