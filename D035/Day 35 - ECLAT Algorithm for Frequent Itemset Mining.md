
# Day 35 - ECLAT Algorithm for Frequent Itemset Mining

## Overview:

This project implements the ECLAT (Equivalence Class Transformation) algorithm for discovering frequent itemsets in transactional datasets.

ECLAT is an association rule mining algorithm that uses a vertical database representation and transaction ID (TID) sets to identify frequently occurring item combinations.

## Objectives

- Understand the ECLAT algorithm.
- Convert transaction data into vertical TID sets.
- Identify frequent itemsets.
- Perform set intersection operations.
- Generate association rules.
- Calculate support, confidence, and lift.

## Technologies Used

- Python
- itertools
- Built-in Python sets

## Dataset

A sample market basket transaction dataset containing products such as:

- Milk
- Bread
- Butter
- Jam

## Algorithm Workflow

1. Load transactional data.
2. Convert transactions into vertical TID sets.
3. Set minimum support threshold.
4. Calculate frequent itemsets using set intersections.
5. Generate association rules.
6. Calculate support, confidence, and lift.
7. Display frequent patterns and rules.

## Important Concepts

### 1. Support

Support measures how frequently an itemset occurs.

Support(X) = Transactions containing X / Total Transactions

### 2. Confidence

Confidence measures the probability of purchasing Y when X is purchased.

Confidence(X -> Y) = Support(X union Y) / Support(X)

### 3. Lift

Lift measures the strength of an association between two itemsets.

Lift(X -> Y) = Confidence(X -> Y) / Support(Y)

Interpretation:

- Lift > 1: Positive association.
- Lift = 1: No association.
- Lift < 1: Negative association.

## How to Run

### Step 1: Clone Repository

```bash
git clone https://github.com/Shambhuraje-96GitH/365ORIENTED.git
```

### Step 2: Navigate to Project

```bash
cd 365ORIENTED/D035
```

### Step 3: Run Python File

```bash
python "Day 35 - ECLAT Algorithm.py"
```

## Expected Output

The program displays:

- Vertical transaction database.
- Frequent itemsets.
- Association rules.
- Support values.
- Confidence values.
- Lift values.
- Final execution summary.

## Applications

- Market Basket Analysis
- Product Recommendation Systems
- Retail Analytics
- Customer Purchasing Behaviour Analysis
- E-commerce Product Bundling

## Advantages of ECLAT

- Uses vertical database representation.
- Efficient set intersection operations.
- Avoids repeated database scans during recursive mining.
- Useful for discovering frequent item combinations.

## Limitations

- TID sets can consume considerable memory for large datasets.
- Performance depends on dataset density.
- Recursive mining may become expensive for large itemsets.

## Conclusion

The ECLAT algorithm demonstrates how vertical transaction databases and set intersections can be used to discover frequent itemsets and association rules.

This project provides practical experience with data mining techniques used in recommendation systems and retail analytics.

---

**Challenge:** 365 Days AIML Engineer Challenge

**Day:** 35

**Topic:** ECLAT Algorithm

**Status:** Completed
