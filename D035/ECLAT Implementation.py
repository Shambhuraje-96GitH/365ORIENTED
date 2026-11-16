
from itertools import combinations

# ==========================================
# DAY 35: ECLAT ALGORITHM
# Frequent Itemset Mining
# ==========================================

# 1. Transaction Dataset
transactions = [
    ["Milk", "Bread", "Butter"],
    ["Bread", "Butter", "Jam"],
    ["Milk", "Bread", "Butter", "Jam"],
    ["Milk", "Bread"],
    ["Bread", "Butter"],
    ["Milk", "Butter"],
    ["Milk", "Bread", "Butter"],
    ["Bread", "Jam"],
    ["Milk", "Bread", "Jam"],
    ["Milk", "Butter", "Jam"]
]

# 2. Configuration
MIN_SUPPORT = 0.3
MIN_CONFIDENCE = 0.5

total_transactions = len(transactions)
min_support_count = int(
    MIN_SUPPORT * total_transactions
)

# 3. Convert dataset into vertical TID sets
vertical_data = {}

for tid, transaction in enumerate(transactions):
    for item in transaction:
        vertical_data.setdefault(item, set()).add(tid)

print("=" * 60)
print("DAY 35 - ECLAT FREQUENT ITEMSET MINING")
print("=" * 60)

print("\nVERTICAL DATABASE (TID SETS)")
print("-" * 40)

for item, tids in sorted(vertical_data.items()):
    print(f"{item}: {sorted(tids)}")

# 4. ECLAT Algorithm
frequent_itemsets = {}

def eclat(prefix, items):
    for index, (item, tids) in enumerate(items):

        support_count = len(tids)
        support = support_count / total_transactions

        if support_count >= min_support_count:

            new_itemset = prefix + (item,)

            frequent_itemsets[new_itemset] = support

            print(
                f"Frequent Itemset: {new_itemset} "
                f"| Support: {support:.2f}"
            )

            suffix = []

            for next_item, next_tids in items[index + 1:]:

                intersection = tids.intersection(next_tids)

                if len(intersection) >= min_support_count:
                    suffix.append((next_item, intersection))

            eclat(new_itemset, suffix)

# 5. Execute ECLAT
sorted_items = sorted(vertical_data.items())

print("\nFREQUENT ITEMSETS")
print("-" * 40)

eclat((), sorted_items)

# 6. Generate Association Rules
print("\nASSOCIATION RULES")
print("-" * 60)

rules = []

for itemset, itemset_support in frequent_itemsets.items():

    if len(itemset) < 2:
        continue

    for size in range(1, len(itemset)):

        for antecedent in combinations(itemset, size):

            consequent = tuple(
                item for item in itemset
                if item not in antecedent
            )

            antecedent_support = frequent_itemsets.get(
                antecedent, 0
            )

            consequent_support = frequent_itemsets.get(
                consequent, 0
            )

            if antecedent_support == 0 or consequent_support == 0:
                continue

            confidence = (
                itemset_support / antecedent_support
            )

            lift = confidence / consequent_support

            if confidence >= MIN_CONFIDENCE:

                rules.append({
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "support": itemset_support,
                    "confidence": confidence,
                    "lift": lift
                })

# 7. Display Rules
if rules:
    rules.sort(
        key=lambda rule: rule["lift"],
        reverse=True
    )

    for rule in rules:
        print(
            f"{rule['antecedent']} -> "
            f"{rule['consequent']}"
        )
        print(
            f"Support: {rule['support']:.2f} | "
            f"Confidence: {rule['confidence']:.2f} | "
            f"Lift: {rule['lift']:.2f}"
        )
        print("-" * 40)
else:
    print("No association rules found.")

# 8. Summary
print("\nFINAL SUMMARY")
print("=" * 60)

print(f"Total Transactions: {total_transactions}")
print(f"Minimum Support: {MIN_SUPPORT}")
print(f"Minimum Confidence: {MIN_CONFIDENCE}")

print(
    f"Total Frequent Itemsets: "
    f"{len(frequent_itemsets)}"
)

print(f"Total Association Rules: {len(rules)}")

print("\nDay 35 ECLAT Implementation Completed!")