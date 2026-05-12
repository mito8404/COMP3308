import csv
# make_folds.py
# This script generates heart-folds.csv automatically when run.

INPUT_FILE = "heart.csv"
OUTPUT_FILE = "heart-folds.csv"
NUM_FOLDS = 10


def main():
    # Read all non-empty lines
    with open(INPUT_FILE, "r") as f:
        rows = [line.strip() for line in f if line.strip()]

    # Remove header if present
    if rows and rows[0].lower().startswith("age"):
        rows = rows[1:]

    # Separate rows by class label
    died_rows = []
    survived_rows = []

    for row in rows:
        label = row.split(",")[-1].strip()
        if label == "died":
            died_rows.append(row)
        elif label == "survived":
            survived_rows.append(row)

    # Create 10 empty folds
    folds = [[] for _ in range(NUM_FOLDS)]

    # Distribute died examples evenly
    for i, row in enumerate(died_rows):
        folds[i % NUM_FOLDS].append(row)

    # Distribute survived examples evenly
    for i, row in enumerate(survived_rows):
        folds[i % NUM_FOLDS].append(row)

    # Write heart-folds.csv
    with open(OUTPUT_FILE, "w") as f:
        for i, fold in enumerate(folds):
            f.write(f"fold{i+1}\n")
            for row in fold:
                f.write(row + "\n")
            if i < NUM_FOLDS - 1:
                f.write("\n")


# Automatically run when the file is executed
main()
