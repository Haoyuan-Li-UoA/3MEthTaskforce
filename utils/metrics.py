import torch
from sklearn.metrics import average_precision_score, roc_auc_score, mean_squared_error, mean_absolute_error


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """

    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """

    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    mse = mean_squared_error(y_true=labels, y_pred=predicts)
    # mae = mean_absolute_error(y_true=labels, y_pred=predicts)
    # roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'mean_squared_error': mse}
