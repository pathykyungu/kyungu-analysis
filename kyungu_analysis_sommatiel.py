import sympy as sp
import numpy as np
import scipy.integrate as integrate

class KyunguSommatiel:
    """
    Deuxième Brique de l'Analyse Sommatielle (Professeur Pathy Kyungu Ngoïe).
    Le Sommatiel [F]_x est la solution de l'équation aux différences finies de Nörlund.
    Le Sommatiel dilaté S(x,s) = [F(sx)]_x constitue le prolongement analytique naturel
    de la somme discrète \sum_{k=1}^{x} F(sk) pour des valeurs réelles ou complexes de x.
    Prolongement analytique sur [0^-, +\infty[ via un contour de Hankel.
    """
    
    def __init__(self):
        pass

    def compute_sommatiel_symbolic(self, F_p, p, x_val, s_val=1):
        """
        Calcule le Sommatiel S(x, s) de manière analytique formelle.
        S(x, s) = \int_{0^-}^\infty \frac{1 - e^{x s t}}{e^{s t} - 1} \mathcal{L}^{-1}{F}(t) dt
        """
        t = sp.Symbol('t', positive=True)
        x = sp.Symbol('x')
        s = sp.Symbol('s', positive=True)
        
        # Inversion de Laplace via la brique 1 ou le moteur formel
        try:
            L_inv_F = sp.inverse_laplace_transform(F_p, p, t)
        except:
            L_inv_F = sp.Function('f')(t)
            
        noyau = (1 - sp.exp(x * s * t)) / (sp.exp(s * t) - 1)
        integrand = noyau * L_inv_F
        
        try:
            sommatiel_expr = sp.integrate(integrand, (t, 0, sp.oo))
            return sommatiel_expr.subs({x: x_val, s: s_val})
        except:
            return sp.Integral(integrand, (t, 0, sp.oo)).subs({x: x_val, s: s_val})

    def compute_sommatiel_numeric(self, f_t_func, x_val, s_val=1, t_max=100):
        """
        Évalue numériquement le Sommatiel S(x, s) = [F(sx)]_x sur [0^-, +\infty[.
        """
        if s_val <= 0:
            raise ValueError("Le paramètre de dilatation s doit être strictement positif.")
            
        def integrand_num(t):
            if t == 0:
                # Prolongement par continuité à l'origine (0^-) : -x * s * f(0) / s = -x * f(0)
                return -x_val * f_t_func(0)
                
            noyau = (1 - np.exp(x_val * s_val * t)) / (np.exp(s_val * t) - 1)
            return noyau * f_t_func(t)
            
        res, _ = integrate.quad(integrand_num, 0, t_max, limit=100)
        return res

    def verify_discrete_sum_match(self, F_func, f_t_func, x_int, s_val=1):
        """
        Vérifie la cohérence du prolongement : 
        S(x, s) doit être exactement égal à la somme discrète de k=1 à x de F(k*s)
        lorsque x est un entier positif.
        """
        if not isinstance(x_int, int) or x_int < 1:
            raise ValueError("Pour une comparaison discrète directe, x doit être un entier >= 1.")
            
        # 1. Calcul par votre formule intégrale continue
        valeur_continue = self.compute_sommatiel_numeric(f_t_func, x_int, s_val)
        
        # 2. Sommation discrète brute de votre théorème : \sum_{k=1}^x F(k*s)
        valeur_discrete = sum(F_func(k * s_val) for k in range(1, x_int + 1))
        
        return {
            'S(x,s) Intégral': valeur_continue,
            'Somme Discrète brute': valeur_discrete,
            'Erreur absolue': abs(valeur_continue - valeur_discrete)
        }
