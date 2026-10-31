
"""
Day 34: FP-Growth Algorithm for Frequent Pattern Mining

Objective:
Discover frequent itemsets and association rules
using the FP-Growth algorithm.

Author: AIML 365-Day Challenge
"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import (
    fpgrowth,
    association_rules
)


# --------------------------------------------------
# 1. Create Transaction Dataset
# --------------------------------------------------

transactions = [
    ["Milk", "Bread", "Butter"],
    ["Bread", "Diapers", "Beer", "Eggs"],
    ["Milk", "Bread", "Diapers", "Beer"],
    ["Milk", "Bread", "Butter"],
    ["Bread", "Butter"],
    ["Milk", "Diapers", "Beer"],
    ["Bread", "Milk", "Butter", "Eggs"],
    ["Diapers", "Beer"],
    ["Milk", "Bread", "Diapers", "Butter"],
    ["Bread", "Milk", "Beer"]
]

print("Original Transactions:")
for index, transaction in enumerate(transactions, 1):
    print(f"Transaction {index}: {transaction}")


# --------------------------------------------------
# 2. Convert Transactions into One-Hot Encoding
# --------------------------------------------------

encoder = TransactionEncoder()

encoded_data = encoder.fit(transactions).transform(
    transactions
)

df = pd.DataFrame(
    encoded_data,
    columns=encoder.columns_
)

print("\nTransaction Matrix:")
print(df)


# --------------------------------------------------
# 3. Apply FP-Growth Algorithm
# --------------------------------------------------

frequent_itemsets = fpgrowth(
    df,
    min_support=0.2,
    use_colnames=True
)

frequent_itemsets = frequent_itemsets.sort_values(
    by="support",
    ascending=False
)

print("\nFrequent Itemsets:")
print(frequent_itemsets.to_string(index=False))


# --------------------------------------------------
# 4. Generate Association Rules
# --------------------------------------------------

if not frequent_itemsets.empty:

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.5
    )

    if not rules.empty:

        rules = rules.sort_values(
            by="lift",
            ascending=False
        )

        print("\nAssociation Rules:")

        display_columns = [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]

        print(
            rules[display_columns].to_string(index=False)
        )

    else:
        print("\nNo association rules found.")

else:
    print("\nNo frequent itemsets found.")


# --------------------------------------------------
# 5. Find Strong Association Rules
# --------------------------------------------------

if not frequent_itemsets.empty and not rules.empty:

    strong_rules = rules[
        (rules["confidence"] >= 0.6) &
        (rules["lift"] > 1)
    ]

    print("\nStrong Association Rules:")

    if not strong_rules.empty:
        print(
            strong_rules[display_columns].to_string(
                index=False
            )
        )
    else:
        print("No strong rules found.")


# --------------------------------------------------
# 6. Save Results
# --------------------------------------------------

frequent_itemsets.to_csv(
    "frequent_itemsets.csv",
    index=False
)

if not frequent_itemsets.empty and not rules.empty:
    rules.to_csv("association_rules.csv", index=False)

print("\nResults saved successfully!")
