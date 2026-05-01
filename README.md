# Bayesian Network — Medical Diagnosis

A single-file Bayesian Network built from scratch in Python. Models the probability of a patient having **Flu or COVID-19** given observed symptoms, using exact inference via Variable Elimination.

---

## Run it

```bash
pip install numpy
python bayesian_network.py
```

---

## What it does

The network reasons about 6 variables:

```
    Flu          COVID
     |  \       / |  \
     |   ▼     ▼  |   ▼
     ▼  Fever Cough ▼  Breathing
    Fatigue        Fatigue
```

| Variable    | Meaning                         |
|-------------|---------------------------------|
| `Flu`       | Patient has influenza (prior 5%)|
| `COVID`     | Patient has COVID-19 (prior 2%) |
| `Fever`     | Caused by Flu or COVID          |
| `Cough`     | Caused by Flu or COVID          |
| `Fatigue`   | Caused by Flu or COVID          |
| `Breathing` | Breathing difficulty — COVID only |

Given any combination of observed symptoms as **evidence**, the network computes the updated posterior probability of each disease.

---

## Sample Output

```
Prior (no symptoms):
  P(COVID=True) = 0.0200

Fever only:
  P(COVID=True) = 0.2305   ↑ from 2%

Fever + Cough + Fatigue + Breathing:
  P(COVID=True) = 0.9292   ← strong evidence
  P(Flu=True)   = 0.1279

Fever + Cough, Breathing=False:
  P(COVID=True) = 0.1515   ↓ negative evidence rules it out
```

Breathing difficulty is the key discriminating symptom — observing it shifts COVID probability from 2% to 93%.

---

## How it works

Three classes, all in one file:

- **`Factor`** — stores a probability table over a set of variables. Supports `restrict` (fix a variable to an observed value), `multiply` (factor product), `marginalise` (sum out a variable), and `normalise`.
- **`BayesianNetwork`** — holds the DAG structure and CPTs. The `query(var, evidence)` method runs Variable Elimination to compute P(var | evidence).
- **`build_network()`** — constructs the medical diagnosis BN with hand-coded CPT values based on domain priors.

### Variable Elimination (in brief)
1. Take all CPT factors
2. Fix any observed evidence by restricting factors
3. For each variable not in the query or evidence: multiply all factors containing it, then sum it out
4. Multiply remaining factors and normalise → posterior distribution.


