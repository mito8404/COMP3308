import csv
import math
from collections import Counter


# =============================================================================
# Helper: read CSV files
# =============================================================================

def _read_rows(filename):
    rows = []

    with open(filename, "r") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue

            row = [value.strip() for value in row]

            # Skip header if present
            if row[0].lower() == "age":
                continue

            rows.append(row)

    return rows


# =============================================================================
# Core KNN
# =============================================================================

def classify_knn(training_filename, testing_filename, k):
    training_data = _read_rows(training_filename)
    testing_data = _read_rows(testing_filename)

    if not training_data or not testing_data:
        return []

    num_attributes = len(training_data[0]) - 1
    predictions = []

    for test_instance in testing_data:
        distances = []

        for row_index, train_instance in enumerate(training_data):
            mismatch_count = 0

            for i in range(num_attributes):
                if test_instance[i] != train_instance[i]:
                    mismatch_count += 1

            distance = math.sqrt(mismatch_count)

            # Store row_index for distance tie-breaking
            distances.append((distance, row_index, train_instance[-1]))

        # Distance tie: lower row index wins
        distances.sort(key=lambda x: (x[0], x[1]))

        neighbours = distances[:k]

        votes = {}
        for _, _, label in neighbours:
            votes[label] = votes.get(label, 0) + 1

        max_votes = max(votes.values())
        candidates = [label for label, count in votes.items() if count == max_votes]

        # Voting tie: predict died
        if len(candidates) == 1:
            predictions.append(candidates[0])
        else:
            predictions.append("died")

    return predictions


# =============================================================================
# Core Decision Tree
# =============================================================================

ATTR_NAMES = [
    "age",
    "anaemia",
    "CPK",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
]


def _entropy(instances):
    total = len(instances)

    if total == 0:
        return 0.0

    counts = {}

    for row in instances:
        label = row[-1]
        counts[label] = counts.get(label, 0) + 1

    entropy_value = 0.0

    for count in counts.values():
        probability = count / total
        if probability > 0:
            entropy_value -= probability * math.log2(probability)

    return entropy_value


def _majority_class(instances):
    counts = {}

    for row in instances:
        label = row[-1]
        counts[label] = counts.get(label, 0) + 1

    if not counts:
        return "died"

    max_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == max_count]

    # Class tie: predict died
    if len(candidates) == 1:
        return candidates[0]

    return "died"


def _information_gain(instances, attr_index):
    total = len(instances)

    if total == 0:
        return 0.0

    parent_entropy = _entropy(instances)

    partitions = {}

    for row in instances:
        value = row[attr_index]

        if value not in partitions:
            partitions[value] = []

        partitions[value].append(row)

    weighted_child_entropy = 0.0

    for subset in partitions.values():
        weighted_child_entropy += (len(subset) / total) * _entropy(subset)

    return parent_entropy - weighted_child_entropy


def _build_tree(instances, available_attrs):
    if not instances:
        return ("leaf", "died")

    classes = set(row[-1] for row in instances)

    if len(classes) == 1:
        return ("leaf", list(classes)[0])

    if not available_attrs:
        return ("leaf", _majority_class(instances))

    best_attr = None
    best_gain = -1.0

    for attr_index in available_attrs:
        gain = _information_gain(instances, attr_index)

        if gain > best_gain:
            best_gain = gain
            best_attr = attr_index

    if best_gain <= 0.0:
        return ("leaf", _majority_class(instances))

    partitions = {}

    for row in instances:
        value = row[best_attr]

        if value not in partitions:
            partitions[value] = []

        partitions[value].append(row)

    remaining_attrs = [attr for attr in available_attrs if attr != best_attr]

    children = {}

    for value, subset in partitions.items():
        children[value] = _build_tree(subset, remaining_attrs)

    default_class = _majority_class(instances)

    return ("node", best_attr, children, default_class)


def _predict_tree(tree, instance):
    if tree[0] == "leaf":
        return tree[1]

    _, attr_index, children, default_class = tree
    value = instance[attr_index]

    if value in children:
        return _predict_tree(children[value], instance)

    return default_class


def classify_dt(training_filename, testing_filename):
    training_data = _read_rows(training_filename)
    testing_data = _read_rows(testing_filename)

    if not training_data or not testing_data:
        return []

    num_attributes = len(training_data[0]) - 1
    available_attrs = list(range(num_attributes))

    tree = _build_tree(training_data, available_attrs)

    predictions = []

    for instance in testing_data:
        predictions.append(_predict_tree(tree, instance))

    return predictions


def print_tree(tree, indent=0):
    prefix = "  " * indent

    if tree[0] == "leaf":
        print(prefix + "-> " + tree[1])
        return

    _, attr_index, children, _ = tree

    if attr_index < len(ATTR_NAMES):
        attr_name = ATTR_NAMES[attr_index]
    else:
        attr_name = str(attr_index)

    for value, subtree in children.items():
        print(prefix + attr_name + " = " + value + ":")
        print_tree(subtree, indent + 1)


def build_and_print_dt(training_filename):
    training_data = _read_rows(training_filename)
    num_attributes = len(training_data[0]) - 1
    available_attrs = list(range(num_attributes))
    tree = _build_tree(training_data, available_attrs)
    print_tree(tree)


# =============================================================================
# KNN+
# Distance-weighted KNN with ordinal-aware distance
# =============================================================================

ORDINAL_MAPS = {
    0: {"[40-50]": 0, "[51-60]": 1, "[61-70]": 2, "70plus": 3},
    2: {"normal": 0, "high": 1, "very-high": 2, "severely-high": 3},
    4: {"low": 0, "mildly-low": 1, "borderline": 2, "normal": 3},
    6: {"low": 0, "normal": 1, "high": 2, "very-high": 3},
    7: {"low": 0, "normal": 1, "high": 2},
    8: {"low": 0, "normal": 1, "high": 2},
}


def _load_labelled_data(filename):
    rows = _read_rows(filename)
    data = []

    for row in rows:
        features = row[:-1]
        label = row[-1]
        data.append((features, label))

    return data


def _ordinal_distance(a, b, num_features, max_ordinal_range):
    total = 0.0

    for i in range(num_features):
        if i in ORDINAL_MAPS:
            mapping = ORDINAL_MAPS[i]
            rank_a = mapping.get(a[i])
            rank_b = mapping.get(b[i])

            if rank_a is not None and rank_b is not None:
                total += (abs(rank_a - rank_b) / max_ordinal_range[i]) ** 2
            else:
                total += 1.0
        else:
            if a[i] == b[i]:
                total += 0.0
            else:
                total += 1.0

    return math.sqrt(total)


def classify_knnplus(training_filename, testing_filename, k):
    training_data = _load_labelled_data(training_filename)
    testing_data = _load_labelled_data(testing_filename)

    if not training_data or not testing_data:
        return []

    num_features = len(training_data[0][0])

    max_ordinal_range = {}

    for feature_index, mapping in ORDINAL_MAPS.items():
        if feature_index < num_features:
            values = list(mapping.values())
            max_ordinal_range[feature_index] = max(values) - min(values)

            if max_ordinal_range[feature_index] == 0:
                max_ordinal_range[feature_index] = 1

    epsilon = 1e-8
    predictions = []

    for test_features, _ in testing_data:
        distances = []

        for row_index, (train_features, train_label) in enumerate(training_data):
            distance = _ordinal_distance(
                test_features,
                train_features,
                num_features,
                max_ordinal_range,
            )

            distances.append((distance, row_index, train_label))

        distances.sort(key=lambda x: (x[0], x[1]))

        neighbours = distances[:k]

        weighted_votes = {}

        for distance, _, label in neighbours:
            weight = 1.0 / (distance + epsilon)
            weighted_votes[label] = weighted_votes.get(label, 0.0) + weight

        max_weight = max(weighted_votes.values())
        candidates = [
            label for label, weight in weighted_votes.items()
            if abs(weight - max_weight) < epsilon
        ]

        if len(candidates) == 1:
            predictions.append(candidates[0])
        else:
            predictions.append(neighbours[0][2])

    return predictions


# =============================================================================
# Decision Tree+
# Gain ratio decision tree with reduced-error pruning
# =============================================================================

class DTNode:
    def __init__(self):
        self.is_leaf = False
        self.label = None
        self.feature_index = None
        self.children = {}
        self.majority_class = "died"


def _label_entropy(labels):
    total = len(labels)

    if total == 0:
        return 0.0

    counts = Counter(labels)
    entropy_value = 0.0

    for count in counts.values():
        probability = count / total
        if probability > 0:
            entropy_value -= probability * math.log2(probability)

    return entropy_value


def _majority_label(label_list):
    if not label_list:
        return "died"

    counts = Counter(label_list)
    max_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == max_count]

    if len(candidates) == 1:
        return candidates[0]

    return "died"


def _gain_and_split_info(data, feature_index):
    labels = [label for _, label in data]
    parent_entropy = _label_entropy(labels)
    total = len(data)

    partitions = {}

    for features, label in data:
        value = features[feature_index]

        if value not in partitions:
            partitions[value] = []

        partitions[value].append(label)

    child_entropy = 0.0
    split_info = 0.0

    for subset in partitions.values():
        p = len(subset) / total

        child_entropy += p * _label_entropy(subset)

        if p > 0:
            split_info -= p * math.log2(p)

    gain = parent_entropy - child_entropy

    return gain, split_info


def _build_tree_plus(data, available_features, depth=0, max_depth=30):
    node = DTNode()

    labels = [label for _, label in data]

    if not labels:
        node.is_leaf = True
        node.label = "died"
        node.majority_class = "died"
        return node

    node.majority_class = _majority_label(labels)

    if len(set(labels)) == 1:
        node.is_leaf = True
        node.label = labels[0]
        return node

    if not available_features or depth >= max_depth:
        node.is_leaf = True
        node.label = node.majority_class
        return node

    gains = {}
    split_infos = {}

    for feature_index in available_features:
        gain, split_info = _gain_and_split_info(data, feature_index)
        gains[feature_index] = gain
        split_infos[feature_index] = split_info

    average_gain = sum(gains.values()) / len(gains)

    best_feature = None
    best_gain_ratio = -1.0

    for feature_index in available_features:
        gain = gains[feature_index]
        split_info = split_infos[feature_index]

        if gain < average_gain:
            continue

        if split_info == 0:
            gain_ratio = 0.0
        else:
            gain_ratio = gain / split_info

        if gain_ratio > best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_feature = feature_index

    if best_feature is None:
        best_feature = max(available_features, key=lambda f: gains[f])

        if gains[best_feature] <= 0.0:
            node.is_leaf = True
            node.label = node.majority_class
            return node

    node.feature_index = best_feature

    partitions = {}

    for features, label in data:
        value = features[best_feature]

        if value not in partitions:
            partitions[value] = []

        partitions[value].append((features, label))

    remaining_features = [
        feature for feature in available_features
        if feature != best_feature
    ]

    for value, subset in partitions.items():
        node.children[value] = _build_tree_plus(
            subset,
            remaining_features,
            depth + 1,
            max_depth,
        )

    if len(node.children) <= 1:
        node.is_leaf = True
        node.label = node.majority_class

    return node


def _predict_tree_plus(node, features):
    if node.is_leaf:
        return node.label

    value = features[node.feature_index]

    if value in node.children:
        return _predict_tree_plus(node.children[value], features)

    return node.majority_class


def _tree_plus_accuracy(root, data):
    if not data:
        return 0.0

    correct = 0

    for features, label in data:
        if _predict_tree_plus(root, features) == label:
            correct += 1

    return correct / len(data)


def _get_internal_nodes(node, nodes=None):
    if nodes is None:
        nodes = []

    if node.is_leaf:
        return nodes

    for child in node.children.values():
        _get_internal_nodes(child, nodes)

    nodes.append(node)

    return nodes


def _prune_tree_plus(root, validation_data):
    if not validation_data:
        return

    changed = True

    while changed:
        changed = False
        current_accuracy = _tree_plus_accuracy(root, validation_data)

        best_node = None
        best_accuracy = current_accuracy

        internal_nodes = _get_internal_nodes(root)

        for node in internal_nodes:
            saved_is_leaf = node.is_leaf
            saved_label = node.label
            saved_children = node.children

            node.is_leaf = True
            node.label = node.majority_class
            node.children = {}

            new_accuracy = _tree_plus_accuracy(root, validation_data)

            if new_accuracy >= best_accuracy:
                best_accuracy = new_accuracy
                best_node = node

            node.is_leaf = saved_is_leaf
            node.label = saved_label
            node.children = saved_children

        if best_node is not None and best_accuracy >= current_accuracy:
            best_node.is_leaf = True
            best_node.label = best_node.majority_class
            best_node.children = {}
            changed = True


def classify_dtplus(training_filename, testing_filename):
    training_data = _load_labelled_data(training_filename)
    testing_data = _load_labelled_data(testing_filename)

    if not training_data or not testing_data:
        return []

    num_features = len(training_data[0][0])
    available_features = list(range(num_features))

    split_index = int(len(training_data) * 0.8)

    if 0 < split_index < len(training_data):
        build_data = training_data[:split_index]
        validation_data = training_data[split_index:]
    else:
        build_data = training_data
        validation_data = []

    root = _build_tree_plus(build_data, available_features)

    if validation_data:
        _prune_tree_plus(root, validation_data)

    predictions = []

    for features, _ in testing_data:
        predictions.append(_predict_tree_plus(root, features))

    return predictions
