"""
Quadrature utilities:
  - cumulant_domain: compute truncated domain I from eq. (22) of Gambaro (2024)
  - make_grid:       uniform evaluation grid on I
  - trapz:           trapezoidal integration wrapper
"""

import numpy as np
from typing import Tuple


def cumulant_domain(cumulants: dict, L: float = 4.0) -> Tuple[float, float]:
    """
    Domain I = [k1 - L*sqrt(k2 + sqrt(k4)),  k1 + L*sqrt(k2 + sqrt(k4))]
    Equation (22) of Gambaro (2024), following Fang & Oosterlee (2009).

    Parameters
    ----------
    cumulants : dict with keys 'k1', 'k2', 'k4' (first, second, fourth cumulant)
    L         : domain widening parameter (paper uses L=4)

    Returns
    -------
    (a, b) : left and right endpoints of I
    """
    k1 = float(cumulants["k1"])
    k2 = float(cumulants["k2"])
    k4 = float(cumulants["k4"])
    half_width = L * np.sqrt(k2 + np.sqrt(abs(k4)))
    return k1 - half_width, k1 + half_width


def clr_domain(log_p_func, cumulants: dict, L_start: float = 4.0,
               clr_tol: float = 10.0, n_pts: int = 2000) -> Tuple[float, float]:
    """
    Restricted domain: shrink the L=4 domain until |clr(p)(x)| < clr_tol everywhere.
    Used for the Heston 'restricted domain' experiments (Figs 7, 11).

    clr(p)(x) = log(p(x)) - E_nu[log(p(X))]
    For the restricted domain we simply find the largest symmetric sub-interval
    [k1-h, k1+h] where log(p(x)) > -clr_tol (ignoring the mean-shift correction,
    which is small; the dominant effect is p(x) -> 0 in the tails).

    la funzione clr_domain non calcola il vero dominio basato sul clr completo, 
    ma utilizza logp(x)>−10 come criterio numerico proxy per individuare una regione in cui la PDF 
    rimane sufficientemente lontana da zero

    Parameters
    ----------
    log_p_func : callable x -> log(p(x)), evaluated on a fine grid
    cumulants  : dict with keys 'k1', 'k2', 'k4'
    L_start    : initial L used to build the test grid
    clr_tol    : tolerance (paper uses 10)
    n_pts      : number of grid points for evaluation

    Returns
    -------
    (a, b) : restricted domain endpoints
    """
    a0, b0 = cumulant_domain(cumulants, L=L_start) #richiama la f precedente per calcolare il dominio di integrazione con i valore default di L = 4.0
    x = np.linspace(a0, b0, n_pts) #crea la griglia di punti di integrazione
    lp = log_p_func(x) #valuta la funzione log(p(x)) sulla griglia di punti 
    """
    la funzione log_p_func è un paramentro di clr_domain, quest'ultima è chiamata in main.py per calcolare il dominio di integrazione (step14)
    solo a quel punto viene creata effettivamente log_p_func come funzione anonima con la log_p_func(x) = log(p(x)). 
    nel concreto questa funzione fa:
    - valuta la densità benchmark COS di Heston nei punti x (usando la funzione caratteristica cf sul dominio pieno L=4)
    - protezione numerica: se la densità è 0 nelle code, il log darebbe -inf; il floor a 10⁻³⁰⁰ lo evita
    - restituisce log p_COS(x) 
    (per il codice guardare main.py)
    l'obiettivo è quello di trovare il sotto intervallo dove viene rispettata la condizione |clr(p)(x)| < 10 ovvero log(p(x)) > -10.
    """
    valid = lp > -clr_tol #restituisce un vettore di booleani dove True se log(p(x)) > -10, False altrimenti
    if not valid.any():
        raise ValueError("No valid domain found with |log(p)| < clr_tol")
    a_restr = x[valid][0] #prende il primo punto valido
    b_restr = x[valid][-1] #prende l'ultimo punto valido
    return float(a_restr), float(b_restr) #restituisce il dominio di integrazione ristretto


def make_grid(a: float, b: float, n: int) -> np.ndarray:
    """Uniform grid of n points on [a, b].""" 
    return np.linspace(a, b, n)


def trapz(f: np.ndarray, x: np.ndarray) -> float:
    """
    Trapezoidal integration of f over x. Questa funzione è usata per calcolare l'integrale della funzione f sui punti x.
    """
    return float(np.trapz(f, x))
