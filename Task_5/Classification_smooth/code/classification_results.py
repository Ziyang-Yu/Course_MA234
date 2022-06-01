def turn_to_labels(dataset):
    dataset_labels = []
    for i in range(0, len(dataset)):
        if dataset[i] > 150.:
            dataset_labels.append(3)
        elif dataset[i] > 35.:
            dataset_labels.append(2)
        else:
            dataset_labels.append(1)
    return dataset_labels

def results(true, pred):
    labels_true = turn_to_labels(true)
    labels_pred = turn_to_labels(pred)
    count = 0
    for i in range(0, len(labels_true)):
        if labels_true[i] == labels_pred[i]:
            count += 1
    return count / len(labels_true)

