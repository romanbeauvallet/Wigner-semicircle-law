#%% Librairies
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad
from tqdm import tqdm

#%% Matrixs
def matNormbis(N):
    M = np.random.randn(N,N)
    for i in range(N):
        M[(i+1):,i] = M[i,(i+1):]
    return M/np.sqrt(N)

def matNorm(N):
    M = np.random.randn(N,N)
    for i in range(N):
        for j in range(i):
            M[i,j] = M[j,i]
    return M/np.sqrt(N)

print(matNorm(5))

def matPois(N):
    M = np.random.poisson(lam = 1, size=(N,N)) -1
    for i in range(N):
        M[(i+1):,i] = M[i,(i+1):]
    return M/np.sqrt(N)

print(matPois(5))

def matUniform(N):
    M = np.random.uniform(low=-1, high=-1, size=(N,N))
    for i in range(N):
        M[(i+1):,i] = M[i,(i+1):]
    return M/np.sqrt(N)

def spectre(M):
    return np.linalg.eigh(M)[0]

print(spectre(matNorm(5)))

#%% Wigmer
def Wigmer(x):
    if -2<=x<=2:
        return np.sqrt(4-x*x)/2/np.pi
    else:   return 0

#%% Histograms
def hist(N,e):
    '''
    N -- la taille matrice aléatoire
    e -- la précision (nombre de bar de l'histogramme)
    '''
    V = spectre(matNorm(N))
    X = np.linspace(-2,2,1000)
    fig, ax = plt.subplots()
    ax.hist(V, bins=e, density='True')
    Y = [Wigmer(X[k]) for k in range(len(X)) ]
    ax.plot(X, Y)
    plt.show()

hist(3000,55)

#%% distance
def distquadhist(M,N,a):
    '''
    M -- une matrice, qui sera la matrice aléatoire
    N -- dimension de la matrice
    a -- puissance tel que le nombre de box de l'histograme est N**a
    '''
    e = int(N**a) #nombre de box de l'histogramme
    V = spectre(M) #spectre de la matrice
    hauteurs, bords = np.histogram(V, bins=e, density=True)
    points = (bords + bords + 4/e)/2 #on prend les points au centre de chaque box
    dist = np.array([hauteurs[k] - Wigmer(points[k]) for k in range(len(hauteurs))]) #on enlève le dernier points qui ne correspond pas au milieu de la dernière box
    return np.mean(dist*dist) #on retourne la variance

print(distquadhist(matNorm(1000),1000,1/4))

def distquadmatNorm(p,N,a):
    '''
    p -- nombre de matrice sur lequel on calcule la variance
    N -- la dimension de la matrice
    a -- puissance tel que le nombre de box de l'histograme est N**a
    '''
    V = []
    for i in range(p):
        M = matNorm(N)
        v = distquadhist(M,N,a)
        V.append(v)
    return np.mean(V)

print(distquadmatNorm(10,1000,1/4))

def distquadmatPois(p,N,a):
    '''
    p -- nombre de matrice sur lequel on calcule la variance
    N -- la dimension de la matrice
    a -- puissance tel que le nombre de box de l'histograme est N**a
    '''
    V = []
    for i in range(p):
        M = matPois(N)
        v = distquadhist(M,N,a)
        V.append(v)
    return np.mean(V)

def distquadquad(M,N,a):
    V = spectre(M)
    e = int(N**a)
    hauteurs, bords = np.histogram(V,bins=e, density=True)
    I = 0
    for i in range(len(hauteurs)):
        def f(x):
            return (Wigmer(x) - hauteurs[i])*(Wigmer(x) - hauteurs[i])
        I += quad(f, bords[i], bords[i+1])[0]
    return I/4

print(distquadquad(matNorm(500),500,.5))

def distquadmatNormbis(p,N,a):
    '''
    p -- nombre de matrice sur lequel on calcule la variance
    N -- la dimension de la matrice
    a -- puissance tel que le nombre de box de l'histograme est N**a
    '''
    V = []
    for i in range(p):
        M = matNorm(N)
        v = distquadquad(M,N,a)
        V.append(v)
    return np.mean(V)

def distquadmatloibis(p,N,a):
    '''
    p -- nombre de matrice sur lequel on calcule la variance
    N -- la dimension de la matrice
    a -- puissance tel que le nombre de box de l'histograme est N**a
    '''
    V = []
    for i in range(p):
        M = matPois(N)
        v = distquadquad(M,N,a)
        V.append(v)
    return np.mean(V)

def erreuralphaNorm(p,N,n):
    '''
    p -- nombre de matrice qu'on moyenne
    N -- dimension de la matrice
    n -- nombre de alpha différent
    '''
    A = np.linspace(0,1,n)
    mat = []
    for i in range(p):
        M = matNorm(N)
        mat.append(M)
    Err = np.zeros((n,p))
    for i in tqdm(range(n)):
        for j in range(p):
            Err[i,j] = distquadquad(mat[j], N, A[i])
    return [A, np.mean(Err, axis=1)]
#%%Graphe alpha
if __name__=="__main__":
    X, Y = erreuralphaNorm(50,500,5000)
    fig, ax = plt.subplots()
    ax.plot(X, Y, label='erreur')
    ax.axvline(x=X[np.argmin(Y)], linestyle='--', color='r', label=r"$\alpha$ minimisant l'erreur : $\alpha_0$")
    ax.set_xlabel(r'précison $\alpha$', fontsize = 25)
    ax.set_ylabel(r'Erreur $({\epsilon_{\alpha}^n})^2$', fontsize = 25)
    ax.legend()
    plt.show()

#%%Graphe log log loi normale
    X = np.arange(3,501,1)
    Y = [distquadmatNormbis(50,X[k],1/2) for k in tqdm(range(len(X)))] #ainsi définit le code n'est pas optimisé car il retire toutes les matrices à chaque fois, à la place on pourrait juste tirer le bon nombre de matrices supplémentaires
    logX = np.log(X)
    logY = np.log(Y)
    Data = stats.linregress(logX, logY)
    Z = Data.slope*logX + Data.intercept
    fig, ax = plt.subplots()
    ax.plot(logX, logY, label="log(erreur) en fonction de log(n)")
    ax.set_xlabel(r"$\log(n)$")
    ax.set_ylabel(r"$\log(({\epsilon_{\alpha}^n})^2)$")
    ax.plot(logX,Z, linestyle="--", color='r', label=f"pente : {Data.slope:.3f}")
    ax.legend()
    plt.show()

#%%Graphe log log Poisson et uniforme
    X = np.arange(3, 501, 1)
    Y = [distquadmatloibis(50,X[k],1/2) for k in tqdm(range(len(X)))]
    logX = np.log(X)
    logY = np.log(Y)
    Data = stats.linregress(logX, logY)
    print(Data.slope)


