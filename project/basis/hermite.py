"""
Hermite polynomial basis (probabilist's convention).

Normalized Hermite polynomials:
    h_j(x) = He_j(x) / sqrt(j!)

where He_j satisfies:
    He_0 = 1,  He_1 = x,  He_{n+1} = x*He_n - n*He_{n-1}

Orthonormality:
    integral h_m(x) h_n(x) omega(x) dx = delta_{m,n}
where omega(x) = (1/sqrt(2*pi))*exp(-x^2/2) is the standard Gaussian weight.

Derivative property:
    h'_j(x) = sqrt(j) * h_{j-1}(x)
"""

import numpy as np
from math import factorial, sqrt


def He_coeffs(n: int) -> np.ndarray: #n: int indica che l'argomento n deve essere un numero intero (grado del polinomio), -> np.ndarray indica che la funzione restituisce un array di numpy.
    """
    la funz. restituisce il polinomio di Hermite He_n(x) come array di coefficienti.
    Ad esempio: He_3(x) = x^3 - 3x verrà visualizzato come [0, -3, 0, 1] = 0 -3*X + 0*X^2 + 1*X^3
    """
    if n == 0:
        return np.array([1.0]) #caso base: He_0(x) = 1
    if n == 1:
        return np.array([0.0, 1.0]) #caso base: He_1(x) = x
    c_prev = np.array([1.0]) #Memorizza il polinomio precedente
    c_curr = np.array([0.0, 1.0]) #Memorizza il polinomio corrente
    for k in range(1, n): #inizia il ciclo for e viene costruito tante basi quanto il grado del polinomio, se n=4 il ciclo esegue k=1,2,3
        #costruendo He_2, He_3 e He_4
        # #vien applicato l'algoritmo di iterazione per i polinomi successivi: He_{k+1}(x) = x * He_k(x) - k * He_{k-1}(x)
        # x * He_k  →  shift coefficients right by 1
        shifted = np.zeros(len(c_curr) + 1) #crea un array di zeri con lunghezza del polinomio corrente + 1. in quanto quando fai x*He_k, stai aumentando il grado del polinomio di 1.
        shifted[1:] = c_curr #copia i coefficienti del polinomio corrente a partire dalla seconda posizione (indice 1), questo è esattamente x*He_k
        c_next = shifted - k * np.pad(c_prev, (0, len(shifted) - len(c_prev))) #questa riga implementa la relazione di ricorrenza: He_{k+1} = x * He_k - k * He_{k-1}
        #per far avvenire la sottrazione fra shifted e il polinomio precedente essi devono avere la stessa lunghezza, per questo viene usato np.pad(c_prev, (0, len(shifted) - len(c_prev))) per aggiungere zeri al polinomio precedente. vengono aggiunti a destra in modo da non cambiare il grado del polinomio. si aggiungono tanti zeri quanti ne mancano per arrivare alla lunghezza di shifted.
        c_prev = c_curr #aggiorna il polinomio precedente con il polinomio corrente
        c_curr = c_next #aggiorna il polinomio corrente con il polinomio successivo
    return c_curr #restituisce il polinomio corrente, che è il polinomio di Hermite He_n(x)


def h_coeffs(n: int) -> np.ndarray:
    """
    Restituisce il vettore dei coefficienti monomiali del polinomio di Hermite
    NORMALIZZATO h_n(x) = He_n(x) / sqrt(n!).

    La normalizzazione garantisce l'ortonormalità rispetto al peso gaussiano:
        integral h_m(x) h_n(x) omega(x) dx = delta_{m,n}

    Ad esempio: h_2(x) = He_2(x)/sqrt(2!) = (x^2 - 1)/sqrt(2) → [-0.707, 0, 0.707]
    """
    return He_coeffs(n) / sqrt(factorial(n))  # divide ogni coefficiente di He_n per sqrt(n!)


def hermite_basis_matrix(N: int) -> np.ndarray:
    """
    Costruisce la matrice H^n di dimensione (N+1) x (N+1) dove la riga i
    contiene i coefficienti monomiali di h_i(x), con zero-padding.

    H_N[i, j] = coefficiente di x^j in h_i(x),   per i,j = 0,...,N.

    Questa è la matrice H^n del paper (eq. 14): ogni riga rappresenta un
    polinomio di Hermite normalizzato scritto nella base monomiale {1, x, x^2, ...}.
    La struttura è lower-triangolare: H[i,j] = 0 per j > i (h_i ha grado i).

    Usata nel cambio di base Q^n = B^n * (H^n)^{-1}  (eq. 14 Gambaro 2024).
    """
    H = np.zeros((N + 1, N + 1)) #crea una matrice di zeri con dimensione (N+1) x (N+1)
    for i in range(N + 1):       #per ogni riga i della matrice, si calcola il vettore dei coefficienti monomiali di h_i(x)
        c = h_coeffs(i)          # vettore coefficienti di h_i, lunghezza i+1
        H[i, :len(c)] = c        # da la dimensione della riga i che deve essere lunga quanto h_i, e copia i coefficienti di h_i in questa riga
    return H


def eval_hermite(x: np.ndarray, N: int) -> np.ndarray: #x vettore dei punti nel quale valutare i polinomi di Hermite, N grado massimo del polinomio
    """
    Valuta tutti i polinomi di Hermite normalizzati h_0,...,h_N in ogni punto di x.
    Restituisce un array di shape (N+1, len(x)): riga k = h_k valutato su tutta la griglia.

    Usa la ricorrenza a tre termini sui VALORI (non sui coefficienti): questo è
    numericamente stabile per qualsiasi N e qualsiasi x, evitando overflow/cancellazione
    che si verificherebbe valutando il polinomio tramite i coefficienti monomiali.

    Ricorrenza normalizzata derivata da He_{k+1} = x*He_k - k*He_{k-1}:
        h_{k+1}(x) = [ x * h_k(x) - sqrt(k) * h_{k-1}(x) ] / sqrt(k+1)

    la differenza con h_coeffs è che qui si valuta il polinomio in ogni punto della griglia x, mentre in h_coeffs restituisce i coefficienti monomiali.
    ogni riga rappresenta un polinomio
    ogni colonna rappresenta un punto della griglia x 

    se N= 2 vuol dire che si hanno 3 polinomi: h_0, h_1, h_2, in base al valore di x si valuta il polinomio h_k(x) per ogni punto della griglia x.
    """
    x = np.asarray(x, dtype=float) #x è un float ovvero un numero reale 
    n = len(x) #n è il numero di punti della griglia x (colonne della matrice P)
    P = np.zeros((N + 1, n)) #crea una matrice di zeri con dimensione (N+1) x n (righe x colonne)
    P[0] = 1.0          # h_0(x) = 1 per ogni x
    if N >= 1:
        P[1] = x        #dato che h_1(x) = x, si assegna il valore di x alla seconda riga della matrice P
    for k in range(1, N): #inizia il ciclo for e si valuta il polinomio h_{k+1}(x) per ogni punto della griglia x
        # Ricorrenza: h_{k+1} = (x * h_k - sqrt(k) * h_{k-1}) / sqrt(k+1)
        P[k + 1] = (x * P[k] - sqrt(k) * P[k - 1]) / sqrt(k + 1) #si valuta il polinomio h_{k+1}(x) per ogni punto della griglia x
    return P #restituisce la matrice P


def gaussian_weight(x: np.ndarray) -> np.ndarray:
    """
    Restituisce il peso gaussiano standard omega(x) = exp(-x^2/2) / sqrt(2*pi).

    Questo è il peso rispetto al quale i polinomi di Hermite normalizzati h_k
    sono ortogonali: integral h_m(x) h_n(x) omega(x) dx = delta_{m,n}.
    Usato come misura di riferimento nu = omega nel framework Bayesian Hilbert Space.
    """
    return np.exp(-x**2 / 2.0) / sqrt(2.0 * np.pi)
