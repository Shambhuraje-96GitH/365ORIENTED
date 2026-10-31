
# Day 34 — FP-Growth Algorithm for Frequent Pattern Mining

## Overview

This project implements the FP-Growth (Frequent Pattern Growth)
algorithm to discover frequent itemsets and association rules
from transactional datasets.

FP-Growth is an efficient frequent-pattern mining algorithm
that uses an FP-Tree structure to avoid generating candidate
itemsets like the Apriori algorithm.

## Objectives

- Understand the FP-Growth algorithm.
- Convert transactional data into one-hot encoding.
- Identify frequent itemsets.
- Generate association rules.
- Evaluate rules using support, confidence, and lift.
- Export mining results into CSV files.

## Technologies Used

- Python
- Pandas
- MLxtend

## Project Structure

D034/
│
├── FP_Growth.py
├── README.md
├── frequent_itemsets.csv
└── association_rules.csv

## Installation

Install the required libraries:

```bash
pip install pandas mlxtend
```

## How to Run

```bash
python FP_Growth.py
```

## Algorithm Workflow

1. Load transactional data.
2. Transform transactions using TransactionEncoder.
3. Generate a one-hot encoded DataFrame.
4. Apply the FP-Growth algorithm.
5. Identify frequent itemsets.
6. Generate association rules.
7. Evaluate association rules.
8. Export results to CSV files.

## Important Evaluation Metrics

### Support

Measures how frequently an itemset appears in the dataset.

Support(A) = Transactions containing A / Total transactions

### Confidence

Measures the probability of purchasing B when A is purchased.

Confidence(A → B) = Support(A ∪ B) / Support(A)

### Lift

Measures the strength of an association between two items.

Lift(A → B) = Confidence(A → B) / Support(B)

Interpretation:

- Lift > 1: Positive association.
- Lift = 1: Independent items.
- Lift < 1: Negative association.

## Applications

- Market Basket Analysis
- E-commerce Product Recommendations
- Customer Purchase Behaviour Analysis
- Retail Inventory Management
- Cross-selling and Upselling
- Recommendation Systems

## Apriori vs FP-Growth

| Feature | Apriori | FP-Growth |
|---|---|---|
| Approach | Candidate generation | FP-Tree |
| Database scans | Multiple | Typically two |
| Candidate generation | Yes | No |
| Efficiency | Can be slower | Often faster |
| Memory | Depends on candidates | FP-Tree storage |

## Learning Outcomes

After completing this project, I understood how
FP-Growth discovers frequent itemsets and how association
rules can reveal purchasing patterns in transactional data.

## Author

AIML 365-Day GitHub Challenge

Day 34 of 365
