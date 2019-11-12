import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torchnet.logger import VisdomPlotLogger
from torch.nn.init import xavier_normal_


def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Parameters
    ----------
    m : array
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables.
    rowvar : bool
        If `rowvar` is True, then each row represents a
        variable, with observations in the columns. Otherwise, the
        relationship is transposed: each column represents a variable,
        while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt)[(None, ) * 2]  # expand it to shape [N,C,H,W]


def batch_cov(tensor):
    """Calculates the covariance of each member of the batch.

    Parameters
    ----------
    tensor : tensor
        A torch tensor of shape [N,C,H,W].

    Returns
    -------
    tensor
        A tensor of shape [N,C,H,W] with covariance of each member.
    """

    output = []
    for i in range(tensor.size(0)):
        output.append(cov(tensor[i].squeeze(dim=0)))
    return torch.cat(output, dim=0)


def weights_init(model):
    """Xavier normal weight initialization for the given model.

    Parameters
    ----------
    model : pytorch model for random weight initialization
    Returns
    -------
    pytorch model with xavier normal initialized weights

    """
    if isinstance(model, nn.Conv2d):
        xavier_normal_(model.weight.data)


def calculate_accuracy(model, data_iterator, key):
    """Calculate the classification accuracy.

    Parameters
    ----------
    model : pytorch object
        A pytorch model.
    data_iterator : pytorch object
        A pytorch dataset.
    key : str
        A key to select which dataset to evaluate

    Returns
    -------
    float
        accuracy of classification for the given key.

    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        total = 0
        length = 0
        for x, y in data_iterator[key]:
            model.eval()
            out_put = model(x.to(device))
            out_put = out_put.cpu().detach()
            total += (out_put.argmax(dim=1) == y.argmax(dim=1)).float().sum()
            length += len(y)
        accuracy = total / length

    return accuracy.numpy()


def classification_accuracy(model, data_iterator):
    """Calculate the classification accuracy of all data_iterators.

    Parameters
    ----------
    model : pytorch object
        A pytorch model.
    data_iterator : dict
        A dictionary with different datasets.

    Returns
    -------
    list
        A dictionary of accuracy for all datasets.

    """
    accuracy = []
    keys = data_iterator.keys()
    for key in keys:
        accuracy.append(calculate_accuracy(model, data_iterator, key))

    return accuracy


def visual_log(title):
    """Return a pytorch tnt visual loggger.

    Parameters
    ----------
    title : str
        A title to describe the logging.

    Returns
    -------
    type
        pytorch visual logger.

    """
    visual_logger = VisdomPlotLogger(
        'line',
        opts=dict(legend=['Training', 'Validation', 'Testing'],
                  xlabel='Epochs',
                  ylabel='Accuracy',
                  title=title))
    return visual_logger


def create_model_info(config, loss_func, accuracy):
    """Create a dictionary of relevant model info.

    Parameters
    ----------
    param : dict
        Any parameter relevant for logging.
    accuracy_log : dict
        A dictionary containing accuracies.

    Returns
    -------
    type
        Description of returned object.

    """
    if accuracy.shape[1] == 2:
        model_info = {
            'training_accuracy': accuracy[:, 0],
            'testing_accuracy': accuracy[:, 1],
            'model_parameters': config,
            'loss function': loss_func
        }
    else:
        model_info = {
            'training_accuracy': accuracy[:, 0],
            'validation_accuracy': accuracy[:, 1],
            'testing_accuracy': accuracy[:, 2],
            'model_parameters': config,
            'loss function': loss_func
        }

    return model_info


def save_trained_pytorch_model(trained_model,
                               trained_model_info,
                               save_path,
                               save_model=True):
    """Save pytorch model and info.

    Parameters
    ----------
    trained_model : pytorch model
    trained_model_info : dict
    save_path : str

    """

    if save_model:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        time_stamp = datetime.now().strftime("%Y_%b_%d_%H_%M")
        torch.save(trained_model, save_path + '/model_' + time_stamp + '.pth')
        torch.save(trained_model_info,
                   save_path + '/model_info_' + time_stamp + '.pth')
        # Save time also
        with open(save_path + '/time.txt', "a") as f:
            f.write(time_stamp + '\n')

    return None


def get_model_path(experiment, model_number):
    """Get all the trained model paths from experiment.

    Parameters
    ----------
    experiment : str
        Which experiment trained models to load.

    Returns
    -------
    model path and model info path

    """

    read_path = str(Path(__file__).parents[2]) + '/models/' + experiment
    with open(read_path + '/time.txt', "r+") as f:
        trained_model = f.readlines()[model_number]
    model_time = trained_model.splitlines()[0]  # remove "\n"
    model_path = str(
        Path(__file__).parents[2]
    ) + '/models/' + experiment + '/model_' + model_time + '.pth'
    model_info_path = str(
        Path(__file__).parents[2]
    ) + '/models/' + experiment + '/model_info_' + model_time + '.pth'

    return model_path, model_info_path


def load_trained_pytorch_model(experiment, model_number):
    """Save pytorch model and info.

    Parameters
    ----------
    trained_model : pytorch model
    trained_model_info : dict
    save_path : str

    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path, model_info_path = get_model_path(experiment, model_number)
    trained_model = torch.load(model_path, map_location=device)

    return trained_model
