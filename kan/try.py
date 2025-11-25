"""
AIM:
Reduce Kannada-English parallel dataset to 250,000 shuffled aligned sentence pairs.
"""

import random

# Your dataset paths
kn_path = "train_kn.txt"
en_path = "train_en.txt"

# Output paths
out_kn = "train_kn_250k.txt"
out_en = "train_en_250k.txt"

LIMIT = 250_000  # 250K

print("Loading files...")

# Read lines
with open(kn_path, "r", encoding="utf-8") as f1:
    kn = [l.strip() for l in f1.readlines()]

with open(en_path, "r", encoding="utf-8") as f2:
    en = [l.strip() for l in f2.readlines()]

print("Kannada lines:", len(kn))
print("English lines:", len(en))

# Ensure aligned length
min_len = min(len(kn), len(en))
kn = kn[:min_len]
en = en[:min_len]

print("Aligned pairs:", min_len)

# Combine + shuffle
pairs = list(zip(kn, en))
random.shuffle(pairs)

# Cut to 250K
pairs = pairs[:LIMIT]

print("Final pairs:", len(pairs))

# Save new files
with open(out_kn, "w", encoding="utf-8") as fkn, \
     open(out_en, "w", encoding="utf-8") as fen:
    for s, t in pairs:
        fkn.write(s + "\n")
        fen.write(t + "\n")

print("\nSaved:")
print("  ->", out_kn)
print("  ->", out_en)
