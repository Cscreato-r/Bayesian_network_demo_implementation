"""
Bayesian Network — Medical Diagnosis (Flu vs COVID)
====================================================
Single-file implementation covering:
  - DAG structure modelling
  - CPT (Conditional Probability Table) representation
  - Exact inference via Variable Elimination

Run: python bayesian_network.py
"""

import itertools
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Factor class
# ─────────────────────────────────────────────────────────────────────────────

class Factor:
    def __init__(self, variables, domains, values):
        self.variables = list(variables)
        self.domains   = {v: list(domains[v]) for v in variables}
        self.values    = np.array(values, dtype=float).reshape(
                            tuple(len(domains[v]) for v in variables))

    def get(self, assignment):
        idx = tuple(self.domains[v].index(assignment[v]) for v in self.variables)
        return float(self.values[idx])

    def restrict(self, var, val):
        axis = self.variables.index(var)
        idx  = self.domains[var].index(val)
        new_vars   = [v for v in self.variables if v != var]
        new_domains = {v: self.domains[v] for v in new_vars}
        return Factor(new_vars, new_domains, np.take(self.values, idx, axis=axis))

    def multiply(self, other):
        all_vars = list(self.variables) + [v for v in other.variables if v not in self.variables]
        all_domains = {v: (self.domains.get(v) or other.domains[v]) for v in all_vars}
        shape  = tuple(len(all_domains[v]) for v in all_vars)
        result = np.zeros(shape)
        for combo in itertools.product(*[all_domains[v] for v in all_vars]):
            assignment = dict(zip(all_vars, combo))
            idx = tuple(all_domains[v].index(assignment[v]) for v in all_vars)
            result[idx] = self.get({v: assignment[v] for v in self.variables}) * \
                          other.get({v: assignment[v] for v in other.variables})
        return Factor(all_vars, all_domains, result)

    def marginalise(self, var):
        axis = self.variables.index(var)
        new_vars = [v for v in self.variables if v != var]
        return Factor(new_vars, {v: self.domains[v] for v in new_vars},
                      self.values.sum(axis=axis))

    def normalise(self):
        return Factor(self.variables, self.domains, self.values / self.values.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian Network class
# ─────────────────────────────────────────────────────────────────────────────

class BayesianNetwork:
    def __init__(self):
        self.variables = []
        self.domains   = {}
        self.parents   = {}
        self.cpts      = {}

    def add_variable(self, name, domain):
        self.variables.append(name)
        self.domains[name]  = list(domain)
        self.parents[name]  = []

    def add_edge(self, parent, child):
        self.parents[child].append(parent)

    def set_cpt(self, variable, values):
        scope   = self.parents[variable] + [variable]
        domains = {v: self.domains[v] for v in scope}
        self.cpts[variable] = Factor(scope, domains, values)

    def query(self, query_var, evidence=None):
        evidence = evidence or {}
        # start with all CPT factors, restrict by evidence
        factors = []
        for v in self.variables:
            f = Factor(self.cpts[v].variables, self.cpts[v].domains,
                       self.cpts[v].values.copy())
            for var, val in evidence.items():
                if var in f.variables:
                    f = f.restrict(var, val)
            factors.append(f)

        # eliminate variables not in query or evidence
        for elim in [v for v in self.variables if v != query_var and v not in evidence]:
            relevant = [f for f in factors if elim in f.variables]
            others   = [f for f in factors if elim not in f.variables]
            product  = relevant[0]
            for f in relevant[1:]:
                product = product.multiply(f)
            others.append(product.marginalise(elim))
            factors = others

        result = factors[0]
        for f in factors[1:]:
            result = result.multiply(f)
        result = result.normalise()
        return {val: result.get({query_var: val}) for val in self.domains[query_var]}


# ─────────────────────────────────────────────────────────────────────────────
# Build the Medical Diagnosis Network
# ─────────────────────────────────────────────────────────────────────────────

def build_network():
    bn = BayesianNetwork()

    # Variables
    bn.add_variable("Flu",       [True, False])
    bn.add_variable("COVID",     [True, False])
    bn.add_variable("Fever",     [True, False])
    bn.add_variable("Cough",     [True, False])
    bn.add_variable("Fatigue",   [True, False])
    bn.add_variable("Breathing", [True, False])

    # Edges
    for symptom in ["Fever", "Cough", "Fatigue"]:
        bn.add_edge("Flu",   symptom)
        bn.add_edge("COVID", symptom)
    bn.add_edge("COVID", "Breathing")

    # CPTs
    bn.set_cpt("Flu",   [0.05, 0.95])           # 5% prevalence
    bn.set_cpt("COVID", [0.02, 0.98])           # 2% prevalence

    # P(Symptom | Flu, COVID) — shape (2,2,2): Flu x COVID x Symptom
    bn.set_cpt("Fever", [
        [[0.95, 0.05], [0.85, 0.15]],   # Flu=T
        [[0.90, 0.10], [0.02, 0.98]],   # Flu=F
    ])
    bn.set_cpt("Cough", [
        [[0.90, 0.10], [0.80, 0.20]],
        [[0.85, 0.15], [0.10, 0.90]],
    ])
    bn.set_cpt("Fatigue", [
        [[0.95, 0.05], [0.90, 0.10]],
        [[0.85, 0.15], [0.05, 0.95]],
    ])

    # P(Breathing | COVID) — shape (2,2): COVID x Breathing
    bn.set_cpt("Breathing", [
        [0.60, 0.40],   # COVID=T
        [0.02, 0.98],   # COVID=F
    ])

    return bn


# ─────────────────────────────────────────────────────────────────────────────
# Run inference queries
# ─────────────────────────────────────────────────────────────────────────────

def print_query(bn, label, query_var, evidence=None):
    result = bn.query(query_var, evidence)
    ev_str = ", ".join(f"{k}={v}" for k, v in evidence.items()) if evidence else "none"
    print(f"\n  {label}")
    print(f"  Evidence : {ev_str}")
    for val, prob in result.items():
        print(f"  P({query_var}={val}) = {prob:.4f}")


if __name__ == "__main__":
    bn = build_network()

    print("=" * 55)
    print("  BAYESIAN NETWORK — MEDICAL DIAGNOSIS")
    print("  (Variable Elimination Inference)")
    print("=" * 55)

    print_query(bn, "Prior — no symptoms",      "Flu")
    print_query(bn, "Prior — no symptoms",      "COVID")
    print_query(bn, "Fever only",               "COVID", {"Fever": True})
    print_query(bn, "Fever + Cough",            "COVID", {"Fever": True, "Cough": True})
    print_query(bn, "Fever + Cough",            "Flu",   {"Fever": True, "Cough": True})
    print_query(bn, "All 4 symptoms",           "COVID", {"Fever": True, "Cough": True, "Fatigue": True, "Breathing": True})
    print_query(bn, "All 4 symptoms",           "Flu",   {"Fever": True, "Cough": True, "Fatigue": True, "Breathing": True})
    print_query(bn, "Fever+Cough, NO Breathing","COVID", {"Fever": True, "Cough": True, "Breathing": False})

    print("\n" + "=" * 55)
