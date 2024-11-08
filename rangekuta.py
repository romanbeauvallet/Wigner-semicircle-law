S = 200  # Nombre d'équations couplées
T = 1000  # Nombre d'itérations
h = 0.1  # Pas de temps
K = 1 #nombre de CI
K = 10 #nombre de CI



# Fonction représentant le système d'équations différentielles couplées
def system(t, N):
    """
    Fonction qui définit le système d'équations différentielles couplées.
    Ici, on a un système générique où chaque composante dépend de l'indice.
    t : temps
    y : vecteur des variables y_i (de taille N)
    Retourne un vecteur de dérivées dy/dt (de taille N).
    """
    dNdt = np.zeros(S)
    # Exemple simple : Couplage entre les équations
    # Vous pouvez modifier ces équations selon vos besoins spécifiques
    for i in range(S):
        cof = 0
        for j in range(S):
            if j == i:
                cof = cof + 0
            else:
                cof = cof + alpha[i,j]*N[j]
        dNdt[i] = N[i]*(1-N[i]) - N[i]*cof  # Couplé avec y[i+1]
            # Couplé avec y[i-1] et y[i+1]
        dNdt[i] = N[i]*(1-N[i]) - N[i]*cof

    return dNdt

@@ -57,7 +43,7 @@

def une_CI():
    # Conditions initiales
    N_init = np.random.rand(S)  # Initialiser les y_i avec des valeurs aléatoires entre 0 et 1
    N_init = np.random.rand(S)  # Initialiser les N_i avec des valeurs aléatoires entre 0 et 1
    t_init = 0  # Temps initial

    # Stockage des résultats
@@ -68,11 +54,6 @@
    for iteration in range(T):
        N = runge_kutta(t, N, h)
        t += h
        # Optionnel : afficher la progression tous les 1000 pas de temps
        #if iteration % 500 == 0:
            #print(f"Iteration {iteration}, Temps t = {t:.2f}")
            #print(N)
    #print("CI terminée")
    return N

#lis = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1, 1.6, 2, 4, 6, 8, 10]
@@ -97,7 +78,7 @@
    #print("Intégration terminée.")
    #print(s)
    var.append(np.mean(Variance))
'''
lis1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]
var1 = [6.380919229242113e-08, 1.5114978426244188e-06, 0.00010615007394414178, 0.0010651338308433644, 0.0031826107934183057, 0.005115017304584216, 0.007893472704669866, 0.023676408446747316, 0.020060778200496406]
@@ -113,14 +94,16 @@
k2 = [c*i*i*i + d*i*i + e*i +f for i in l2]
x = 0.7439
y = 0.017
'''
#La partie du dessus sert à déterminer le comportement limite

plt.tick_params(axis='both', which='major', labelsize=16)
plt.plot(lis, var, marker ='x', color = 'black', linestyle='None', markersize=8, label = r'$\sqrt{\mathbb{V}~(\sigma)}$')
plt.plot(l1, k1, '--', label = r'fit polynomial $\sigma < \sigma_c$')
plt.plot(l2, k2, '--', label = r'fit polynomial $\sigma > \sigma_c$')
#plt.plot(l1, k1, '--', label = r'fit polynomial $\sigma < \sigma_c$')
#plt.plot(l2, k2, '--', label = r'fit polynomial $\sigma > \sigma_c$')
plt.plot(x,y, marker = '.', color = 'blue', markersize = 20, label = r'$\sigma_c = 0.74$')
plt.text(x -0.05, 0.09, r'$\left(\sqrt{\mathbb{V}~(\sigma_c)},~\sigma_c\right)$', ha='center', fontsize=18, color='blue')
plt.xlabel(r'$\sigma$', fontsize = 20)
plt.ylabel(r'$\sqrt{\mathbb{V}~(\sigma)}$', fontsize = 20)
plt.legend(fontsize = 20)
plt.show()