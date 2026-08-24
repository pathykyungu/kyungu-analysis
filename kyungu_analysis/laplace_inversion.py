import sympy as sp

class KyunguLaplaceInversion:
    """
    Première Brique de l'Analyse Sommatielle (Professeur Pathy Kyungu Ngoïe).
    Formule unifiée d'inversion de la transformée de Laplace au sens des distributions.
    Version 12 (Corrigée avec enveloppe e^{bt} globale et condensation en cosh).
    """
    
    def __init__(self):
        pass

    def invert_laplace_general(self, F_p, p, t, a=0, b=0, c=1, num_terms=6):
        """
        Formule générale unifiée de Kyungu avec paramètres de réglage (a, b, c).
        
        Considère la fonction auxiliaire phi(x) = F(b + c/(x-a)).
        Développe phi(x) au voisinage de x = a sous la forme \sum c_k (x-a)^{\alpha_k}.
        Reconstruit l'inverse au sens des distributions sous l'enveloppe e^{bt}.
        """
        x = sp.Symbol('x', positive=True)
        
        # 1. Construction de la fonction auxiliaire phi(x)
        argument_p = b + (c / (x - a))
        phi_x = F_p.subs(p, argument_p)
        
        # 2. Développement au voisinage de x = a via les séries de Puiseux/Laurent
        try:
            serie = sp.series(phi_x, x, a, num_terms).removeO()
        except Exception as e:
            raise ValueError(f"Impossible de développer la fonction auxiliaire au voisinage de x={a}: {e}")
            
        # 3. Extraction des coefficients c_k et exposants alpha_k
        termes = sp.Add.make_args(serie)
        
        crochet_unifie = 0  # Accumulateur des 3 composantes
        partie_reguliere = 0
        termes_impulsionnels = [] 
        termes_distributionnels = [] 
        
        # Dictionnaires pour stocker les coefficients par ordre de dérivation pour la condensation
        coefficients_dirac = {}
        
        for terme in termes:
            # Séparation robuste du coefficient indépendant de x
            coeff, reste = terme.as_coeff_mul(x)
            
            if reste == (): # Terme constant (alpha_k = 0)
                alpha_k = sp.Integer(0)
                c_k = coeff
            else:
                expr_x = reste[0]
                if expr_x.is_Pow and (expr_x.base == (x - a) or expr_x.base == x):
                    alpha_k = expr_x.exp
                    c_k = coeff
                elif expr_x == (x - a) or expr_x == x:
                    alpha_k = sp.Integer(1)
                    c_k = coeff
                else:
                    continue
                
            # 4. Classification selon le théorème maître (v12)
            
            # Cas 1 : Termes impulsionnels (alpha_k == 0) -> c_k * delta(t)
            if alpha_k == 0:
                coefficients_dirac[0] = c_k
                termes_impulsionnels.append({
                    'coeff_analytique': c_k,
                    'expression_latex': f"{sp.latex(c_k)} \\cdot \\delta(t)"
                })
                
            # Cas 2 : Termes distributionnels (alpha_k = -m, m entier positif) -> c_k * c^{-m} * delta^(m)(t)
            elif alpha_k.is_integer and alpha_k < 0:
                m = -int(alpha_k)
                coeff_dist = c_k * (c ** (-m))
                coefficients_dirac[m] = coeff_dist
                termes_distributionnels.append({
                    'ordre_m': m,
                    'coeff_analytique': coeff_dist,
                    'expression_latex': f"{sp.latex(coeff_dist)} \\cdot \\delta^{{({m})}}(t)"
                })
                
            # Cas 3 : Partie régulière (alpha_k non entier négatif ou nul)
            else:
                terme_reg = (c_k * (c ** alpha_k) / sp.gamma(alpha_k)) * (t ** (alpha_k - 1))
                partie_reguliere += terme_reg

        # 5. Algorithme Avancé de Condensation Opératorielle (Reconnaissance cosh(sqrt(a * d/dt)))
        # On vérifie si la structure des coefficients suit exactement (-1)^n * a^n / (2n)!
        is_cosh_pattern = False
        a_detected = None
        
        # Il faut au moins le terme d'ordre 0 et d'ordre 1 pour tenter l'identification de la signature
        if 0 in coefficients_dirac and 1 in coefficients_dirac:
            c0 = coefficients_dirac[0]
            c1 = coefficients_dirac[1]
            
            # Si c0 = 1 et c1 = -a, alors le paramètre candidat de l'opérateur est a = -c1
            a_candidate = -c1
            
            # Validation des termes d'ordres supérieurs présents
            is_cosh_pattern = True
            for m, coeff in coefficients_dirac.items():
                if m % 2 == 1: # Les ordres impairs différents de 1 doivent être nuls dans cos(sqrt(ap))
                    if m != 1 and coeff != 0:
                        is_cosh_pattern = False
                        break
                else: # Les ordres pairs m = 2n doivent valoir c_2n = (-1)^n * a^n / (2n)!
                    n = m // 2
                    expected_coeff = ((-1)**n * (a_candidate**n)) / sp.factorial(2*n)
                    if sp.simplify(coeff - expected_coeff) != 0:
                        is_cosh_pattern = False
                        break
            
            if is_cosh_pattern:
                a_detected = a_candidate

        # Reconstruction finale du crochet unifié selon la condensation circulaire exact
        if is_cosh_pattern and a_detected is not None:
            # Définition symbolique de l'opérateur d/dt au sens de Kyungu
            d_dt = sp.Symbol('d/dt')
            # CORRECTION : C'est le cosinus circulaire qui porte la série alternée (-1)^n
            op_cos = sp.cos(sp.sqrt(a_detected * d_dt))
            crochet_unifie = op_cos * sp.DiracDelta(t) + partie_reguliere
            forme_condensee_active = True
        else:
            # Construction standard si la série ne correspond pas à un cosh pur
            for m, coeff in coefficients_dirac.items():
                if m == 0:
                    crochet_unifie += coeff * sp.DiracDelta(t)
                else:
                    crochet_unifie += coeff * sp.DiracDelta(t, m)
            crochet_unifie += partie_reguliere
            forme_condensee_active = False

        # RETOUCHE MAJEURE (v12) : L'enveloppe e^{bt} globale
        solution_temporelle_finale = sp.exp(b * t) * crochet_unifie

        return {
            'solution_complete': solution_temporelle_finale,
            'partie_reguliere_brute': sp.simplify(partie_reguliere),
            'termes_impulsionnels': termes_impulsionnels,
            'termes_distributionnels': termes_distributionnels,
            'forme_condensee_active': forme_condensee_active
        }

    def invert_laplace_simple(self, F_p, p, t, num_terms=6):
        """
        Cas particulier : phi(x) = F(1/x), développement autour de x = 0.
        """
        return self.invert_laplace_general(F_p, p, t, a=0, b=0, c=1, num_terms=num_terms)

# =====================================================================
# SCRIPT D'ÉVALUATION ET DE VÉRIFICATION SUR MACHINE
# =====================================================================
if __name__ == "__main__":
    p, t = sp.symbols('p t', positive=True)
    a_param = sp.Symbol('a', positive=True)
    
    inverser = KyunguLaplaceInversion()
    
    print("=" * 70)
    print("VÉRIFICATION DU CAS : F(p) = cos(sqrt(a*p))")
    print("=" * 70)
    
    F_cos = sp.cos(sp.sqrt(a_param * p))
    # On pousse à 8 termes pour laisser l'algorithme analyser la suite topologique
    resultat = inverser.invert_laplace_simple(F_cos, p, t, num_terms=8)
    
    print(f"[+] Forme condensée en cosh détectée ? : {resultat['forme_condensee_active']}")
    print(f"[+] Solution Temporelle Unifiée Générée :\n    f(t) = {resultat['solution_complete']}")
    print("=" * 70)
