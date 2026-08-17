import sympy as sp
import numpy as np
import scipy.integrate as integrate

class KyunguSommatiel:
    """
    Deuxième Brique de l'Analyse Sommatielle (Professeur Pathy Kyungu Ngoïe).
    Le Sommatiel [F]_x est la solution de l'équation aux différences finies de Nörlund.
    Le Sommatiel dilaté S(x,s) = [F(sx)]_x constitue le prolongement analytique naturel
    de la somme discrète \sum_{k=1}^{x} F(sk) pour des valeurs réelles ou complexes de x.
    
    Représentation intégrale officielle sur la variété topologique \Gamma :
    S(x, s) = \int_{\Gamma} \frac{1 - e^{x s t}}{e^{s t} - 1} \mathcal{L}^{-1}{F}(t) dt
    
    Où \Gamma est la même variété topologique que l'Hypersommatiel : \Gamma = \mathcal{H} \cup \bigcup_{k} \gamma_k
    - \mathcal{H} : Contour de Hankel enveloppant l'axe réel et l'origine (0^-).
    - \gamma_k  : Micro-contours isolés entourant les singularités distributionnelles de Dirac.
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

    def compute_sommatiel_on_variety(self, f_t_complex_func, x_val, s_val, singularites_dirac=None, R_micro=0.01, t_max=50):
        """
        Évalue le Sommatiel dans le plan complexe par intégration sur la variété topologique 
        unifiée \Gamma = \mathcal{H} \cup \bigcup \gamma_k pour gérer holomorphement les coupures et les Dirac.
        """
        if singularites_dirac is None:
            singularites_dirac = []

        def integrand_fondamental(t_c):
            return ((1 - np.exp(x_val * s_val * t_c)) / (np.exp(s_val * t_c) - 1)) * f_t_complex_func(t_c)

        # 1. COMPOSANTE : Contour de Hankel (\mathcal{H})
        I_h_top_r, _ = integrate.quad(lambda u: np.real(integrand_fondamental(u + 1j * R_micro)), t_max, 0)
        I_h_top_i, _ = integrate.quad(lambda u: np.imag(integrand_fondamental(u + 1j * R_micro)), t_max, 0)
        I_h_top = I_h_top_r + 1j * I_h_top_i

        def integrand_cercle_origine(theta):
            t_c = R_micro * np.exp(1j * theta)
            dt_c = 1j * R_micro * np.exp(1j * theta)
            return integrand_fondamental(t_c) * dt_c
            
        I_h_circle_r, _ = integrate.quad(lambda th: np.real(integrand_cercle_origine(th)), np.pi/2, -np.pi/2)
        I_h_circle_i, _ = integrate.quad(lambda th: np.imag(integrand_cercle_origine(th)), np.pi/2, -np.pi/2)
        I_h_circle = I_h_circle_r + 1j * I_h_circle_i

        I_h_bot_r, _ = integrate.quad(lambda u: np.real(integrand_fondamental(u - 1j * R_micro)), 0, t_max)
        I_h_bot_i, _ = integrate.quad(lambda u: np.imag(integrand_fondamental(u - 1j * R_micro)), 0, t_max)
        I_h_bot = I_h_bot_r + 1j * I_h_bot_i

        contribution_hankel = I_h_top + I_h_circle + I_h_bot

        # 2. COMPOSANTE : Famille de micro-contours (\gamma_k)
        contribution_micro_contours = 0j
        for t_k in singularites_dirac:
            def integrand_lacet(theta):
                t_c = t_k + R_micro * np.exp(1j * theta)
                dt_c = 1j * R_micro * np.exp(1j * theta)
                return integrand_fondamental(t_c) * dt_c
                
            I_k_r, _ = integrate.quad(lambda th: np.real(integrand_lacet(th)), 0, 2 * np.pi)
            I_k_i, _ = integrate.quad(lambda th: np.imag(integrand_lacet(th)), 0, 2 * np.pi)
            contribution_micro_contours += (I_k_r + 1j * I_k_i)

        return contribution_hankel + contribution_micro_contours

    def verify_discrete_sum_match(self, F_func, f_t_func, x_int, s_val=1):
        """
        Vérifie la cohérence du prolongement : 
        S(x, s) doit être exactement égal à la somme discrète de k=1 à x de F(k*s)
        lorsque x est un entier positif.
        """
        if not isinstance(x_int, int) or x_int < 1:
            raise ValueError("Pour une comparaison discrète directe, x doit être un entier >= 1.")
            
        valeur_continue = self.compute_sommatiel_numeric(f_t_func, x_int, s_val)
        valeur_discrete = sum(F_func(k * s_val) for k in range(1, x_int + 1))
        
        return {
            'S(x,s) Intégral': valeur_continue,
            'Somme Discrète brute': valeur_discrete,
            'Erreur absolue': abs(valeur_continue - valeur_discrete)
        }
