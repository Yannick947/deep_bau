import os
import matplotlib.pyplot as plt


def plot_history(history: dict, logging_path: str, show: bool = True):
    # summarize history for accuracy
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Model Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if show:
        plt.show()
    plt.savefig(os.path.join(logging_path, 'mae.png'))
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if show:
        plt.show()
    plt.savefig(os.path.join(logging_path, 'loss.png'))
    plt.close()
