import sympy as sp

class KyunguLaplaceInversion:
    """
    Première Brique de l'Analyse Sommatielle (Professeur Pathy Kyungu Ngoïe).
    Formule unifiée d'inversion de la transformée de Laplace au sens des distributions.
    """
    
    def __init__(self):
        pass

    def invert_laplace_general(self, F_p, p, t, a=0, b=0, c=1, num_terms=6):
        """
        Formule générale unifiée de Kyungu avec paramètres de réglage (a, b, c).
        
        Considère la fonction auxiliaire phi(x) = F(b + c/(x-a)).
        Développe phi(x) au voisinage de x = a sous la forme \sum c_k (x-a)^{\alpha_k}.
        Reconstruit l'inverse au sens des distributions.
        """
        x = sp.Symbol('x', Resource=True)
        
        # 1. Construction de la fonction auxiliaire phi(x)
        argument_p = b + (c / (x - a))
        phi_x = F_p.subs(p, argument_p)
        
        # 2. Développement au voisinage de x = a
        # On effectue un développement de Taylor/Laurent automatique via SymPy
        try:
            serie = sp.series(phi_x, x, a, num_terms).removeO()
        except Exception as e:
            raise ValueError(f"Impossible de développer la fonction auxiliaire au voisinage de x={a}: {e}")
            
        # 3. Extraction des coefficients c_k et exposants alpha_k
        # On réécrit par rapport à (x-a)
        termes = sp.Add.make_args(serie)
        
        partie_reguliere = 0
        termes_impulsionnels = [] # Pour alpha_k == 0 (c_k * delta(t))
        termes_distributionnels = [] # Pour alpha_k = -m (c_k * c^{-m} * delta^(m)(t))
        
        for terme in termes:
            # Isoler le coefficient et la puissance de (x-a)
            coeff = terme.as_independent(x)[0]
            reste = terme.as_independent(x)[1]
            
            if reste == 1: # Terme constant (alpha_k = 0)
                alpha_k = sp.Integer(0)
                c_k = coeff
            elif reste.is_Pow and reste.base == (x - a):
                alpha_k = reste.exp
                c_k = coeff
            elif reste == (x - a):
                alpha_k = sp.Integer(1)
                c_k = coeff
            else:
                # Gestion des cas complexes ou non-standards
                continue
                
            # 4. Classification selon votre théorème maître
            
            # Cas 1 : Termes impulsionnels (alpha_k == 0) -> c_k * delta(t)
            if alpha_k == 0:
                termes_impulsionnels.append({
                    'coeff_analytique': c_k,
                    'expression_latex': f"{sp.latex(c_k)} \\cdot \\delta(t)"
                })
                
            # Cas 2 : Termes distributionnels (alpha_k = -m, m entier positif) -> c_k * c^{-m} * delta^(m)(t)
            elif alpha_k.is_integer and alpha_k < 0:
                m = -int(alpha_k)
                coeff_dist = c_k * (c ** (-m))
                termes_distributionnels.append({
                    'ordre_m': m,
                    'coeff_analytique': coeff_dist,
                    'expression_latex': f"{sp.latex(coeff_dist)} \\cdot \\delta^{{({m})}}(t)"
                })
                
            # Cas 3 : Partie régulière (alpha_k non entier négatif ou nul)
            else:
                # Formule maîtresse : (c_k * c^{\alpha_k} / Gamma(\alpha_k)) * t^{\alpha_k-1} * e^{bt}
                terme_reg = (c_k * (c ** alpha_k) / sp.gamma(alpha_k)) * (t ** (alpha_k - 1)) * sp.exp(b * t)
                partie_reguliere += terme_reg

        return {
            'partie_reguliere': sp.simplify(partie_reguliere),
            'termes_impulsionnels': termes_impulsionnels,
            'termes_distributionnels': termes_distributionnels
        }

    def invert_laplace_simple(self, F_p, p, t, num_terms=6):
        """
        Cas particulier (Section 2.1 & 2.3 de vos recherches) : phi(x) = F(1/x)
        Développement autour de x = 0.
        """
        return self.invert_laplace_general(F_p, p, t, a=0, b=0, c=1, num_terms=num_terms)
