import kv_bottleneck_experiments.utils.base as base_utils
import torch
import wandb


def evaluate_accuracy(
    model,
    dataloader,
    train_step,
    epoch,
    args,
    prefix="",
    reference_accuracy=None,
    current_best_accuracy=None,
    dataset="test",
    wandb_log=True,
    criterion=None,
):
    if prefix != "":
        prefix = prefix + "/"

    num_classes = base_utils.get_class_nums(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct_pred = [0] * num_classes
    total_pred = [0] * num_classes
    with torch.no_grad():
        correct = 0
        total = 0
        loss = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            if criterion is not None:
                batch_loss = criterion(outputs, labels)
                loss += batch_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if not args.no_per_class_acc:
                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        correct_pred[label] += 1
                    total_pred[label] += 1

        acc1 = 100 * float(sum([correct_pred[i]/total_pred[i] for i in range(num_classes)])/num_classes)
        log_dict = {
            prefix + f"{dataset}_accuracy": acc1,
            "epochs": epoch,
            "train_step": train_step,
        }

        if criterion is not None:
            log_dict.update({prefix + f"{dataset}_loss": float(loss)})
        if current_best_accuracy is not None:
            log_dict.update({prefix + f"best_{dataset}_accuracy": current_best_accuracy})
        if reference_accuracy is not None:
            log_dict[prefix + f"adapt_accuracy_gain_{dataset}"] = acc1 - reference_accuracy
        if not args.no_per_class_acc:
            for class_label, correct_count in enumerate(correct_pred):
                accuracy = 100 * float(correct_count) / total_pred[class_label]
                log_dict[f"{prefix}{dataset}_accuracy_{str(class_label)}"] = accuracy
        if wandb_log:
            wandb.log(log_dict)

    return acc1
