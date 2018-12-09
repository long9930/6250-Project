import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, loss_fig="Loss.png", accuracy_fig="accuracy.png"):
    plt.plot(np.arange(len(train_losses)), train_losses, label='Training loss')
    plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(loc="best")
    plt.savefig(loss_fig)
    plt.show()


    plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Train Accuracy')
    plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(loc="best")
    plt.savefig(accuracy_fig)
    plt.show()


def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
    y_label=[i[0] for i in results]
    y_pred= [i[1] for i in results]
    results=confusion_matrix(y_label,y_pred)
    results=results.astype("float")/results.sum(axis=1)[:,np.newaxis]
    plt.imshow(results,interpolation='nearest',cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks=np.arange(len(class_names))
    plt.xticks(tick_marks,class_names,rotation=45)
    plt.yticks(tick_marks,class_names)        
    
    fmt='.2f' 
    thresh=results.max()/2.
    for i,j in itertools.product(range(results.shape[0]),range(results.shape[1])):
        plt.text(j, i, format(results[i,j],fmt),
                 horizontalalignment="center",
                 color="white" if results[i,j]>thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")

    
