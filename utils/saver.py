import numpy as np
import torch


def save_model(model, path):
    return torch.save(model, path)


def save_model_if_best(test_acc, model, path, printf=print, show_acc=True):
    if test_acc[-1] < 0.4:
        return
    epoch = len(test_acc) - 1
    best_epoch = np.argmax(test_acc)

    if epoch == best_epoch:
        if show_acc:
            torch.save(model, path.replace('best', f'{test_acc[-1]:.4f}'))
        else:
            torch.save(model, path)
        printf(f'best saved: {path}')

    else:
        print(f'best:{best_epoch}, cur:{epoch}')


def load_model(path):
    return torch.load(path)
