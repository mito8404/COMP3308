import os
from program import classify_knn, classify_dt, classify_knnplus, classify_dtplus

FOLDS_FILE = "heart-folds.csv"


def read_folds(filename):
    folds = []
    current_fold = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("fold"):
                if current_fold:
                    folds.append(current_fold)
                    current_fold = []

            elif line:
                current_fold.append(line)

        if current_fold:
            folds.append(current_fold)

    return folds


def write_temp_file(filename, rows):
    with open(filename, "w") as f:
        for row in rows:
            f.write(row + "\n")


def calculate_metrics(actual, predicted, positive_class="died"):
    tp = fp = tn = fn = 0

    for a, p in zip(actual, predicted):
        if a == positive_class and p == positive_class:
            tp += 1
        elif a != positive_class and p == positive_class:
            fp += 1
        elif a != positive_class and p != positive_class:
            tn += 1
        elif a == positive_class and p != positive_class:
            fn += 1

    accuracy = (tp + tn) / len(actual)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return accuracy, precision, recall, f1, tp, fp, tn, fn


def evaluate_classifier(name, classifier, folds, k=None):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(len(folds)):
        testing_rows = folds[i]

        training_rows = []
        for j in range(len(folds)):
            if j != i:
                training_rows.extend(folds[j])

        write_temp_file("temp_train.csv", training_rows)
        write_temp_file("temp_test.csv", testing_rows)

        if k is None:
            predictions = classifier("temp_train.csv", "temp_test.csv")
        else:
            predictions = classifier("temp_train.csv", "temp_test.csv", k)

        actual = [row.split(",")[-1].strip() for row in testing_rows]

        accuracy, precision, recall, f1, tp, fp, tn, fn = calculate_metrics(
            actual, predictions
        )

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        print(f"{name} Fold {i + 1}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")

    print()
    print(f"===== {name} Average Results =====")
    print(f"Accuracy:  {sum(accuracies) / len(accuracies):.4f}")
    print(f"Precision: {sum(precisions) / len(precisions):.4f}")
    print(f"Recall:    {sum(recalls) / len(recalls):.4f}")
    print(f"F1-score:  {sum(f1_scores) / len(f1_scores):.4f}")
    print()

    os.remove("temp_train.csv")
    os.remove("temp_test.csv")


def main():
    folds = read_folds(FOLDS_FILE)

    evaluate_classifier("KNN", classify_knn, folds, k=3)
    evaluate_classifier("Decision Tree", classify_dt, folds)

    evaluate_classifier("KNN+", classify_knnplus, folds, k=3)
    evaluate_classifier("Decision Tree+", classify_dtplus, folds)


main()
