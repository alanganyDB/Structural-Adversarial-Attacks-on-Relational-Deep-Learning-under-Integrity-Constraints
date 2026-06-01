# Adversarial Structural Attacks on Relational Graph Neural Networks

This notebook investigates adversarial attacks on relational databases represented as heterogeneous graphs.

The goal is to evaluate how structural perturbations of foreign-key relationships affect the performance of Graph Neural Networks (GNNs) trained on relational prediction tasks.

---

## Dataset

We use the **RelBench** benchmark and, in particular, the **rel-f1** dataset.

The relational database is transformed into a heterogeneous graph where:

- Tables become node types.
- Foreign-key dependencies become edge types.
- Reverse relations are automatically added.
- Node attributes are encoded using TensorFrame and RelBench utilities.

Examples of node types include:

- `drivers`
- `races`
- `results`
- `qualifying`
- `constructors`
- `standings`
- `circuits`

---

## Prediction Tasks

The notebook supports both classification and regression tasks:

### Classification

- `driver-dnf`
    - Predict whether a driver fails to finish a race.

- `driver-top3`
    - Predict whether a driver finishes in the top 3.

### Regression

- `driver-position`
    - Predict the final race position.

- `qualifying-position`
    - Predict the qualifying position.

---

## Model Architecture

The predictive model follows the standard RelBench pipeline:

1. **HeteroEncoder**
    - Encodes tabular attributes for each node type.

2. **HeteroTemporalEncoder**
    - Encodes temporal information.

3. **HeteroGraphSAGE**
    - Performs message passing over the heterogeneous graph.

4. **Prediction Head**
    - Produces task-specific outputs.

An attackable version of the model is also constructed to enable differentiable structural perturbation analysis.

---

## Structural Attack Setting

The attacks operate on foreign-key relationships.

Each perturbation corresponds to a valid rewiring:

- remove an existing FK → PK relation,
- connect the child node to another compatible parent,
- preserve database consistency,
- preserve foreign-key semantics.

Example:

```text
results.driverId = Hamilton
          ↓
results.driverId = Verstappen
```

The resulting graph remains structurally valid while altering the information available to the GNN.

---

## Attack Strategies

Several candidate-selection strategies are evaluated:

### Gradient-based

- Raw gradient scores
- Global Z-score normalization
- Relation-wise Min-Max normalization
- Robust Z-score normalization

### Exact Reranking

A shortlist of candidates is generated using gradients and then reranked using the true downstream metric.

### Random Baselines

- Pure random rewiring
- Random shortlist + exact reranking

---

## Evaluation Protocol

For each validation batch:

1. Compute clean model performance.
2. Generate candidate rewirings.
3. Rank candidates using different scoring strategies.
4. Apply perturbations under multiple attack budgets.
5. Evaluate attacked performance.
6. Measure attack diversity.
7. Record computational cost.

---

## Metrics

### Classification

- Accuracy

### Regression

- Mean Absolute Error (MAE)

The attack objective is to maximize performance degradation.

---

## Outputs

The notebook generates three main result tables:

### Attack Effectiveness

```python
rows
```

Contains:

- clean performance,
- attacked performance,
- performance degradation.

### Diversity Statistics

```python
diversity_rows
```

Contains:

- relation coverage,
- number of relations affected,
- perturbation concentration,
- unique child statistics.

### Runtime Measurements

```python
time_rows
```

Contains:

- gradient scoring time,
- exact reranking time,
- attack application time.

---

## Research Goal

The ultimate objective is to understand whether first-order gradient information can be used to efficiently identify harmful structural perturbations in relational databases and whether such attacks outperform random baselines under realistic database constraints.