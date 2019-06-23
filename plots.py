import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.model_selection import learning_curve
# from sklearn.model_selection import ShuffleSplit

def print_confusion_matrix(confusion_matrix, labels, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)

    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 16}, cmap='Blues', ax=ax)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
    heatmap.yaxis.set_ticks_position('left')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, fontsize=14)
    heatmap.xaxis.set_ticks_position('top')

    ax.set_ylabel('Predicted Class', fontsize=14)
    ax.set_xlabel('Actual Class', fontsize=14)

    ax.xaxis.set_label_position('top')

    plt.show();
    
    
def print_confusion_matrixes(cm_list, labels, figsize = (10,7), fontsize=14):
    if(len(cm_list)) == 1:
        print_confusion_matrix(cm_list[1], labels, figsize = (10,7), fontsize=14)
        return
    
    fig, ax = plt.subplots(1, len(cm_list), constrained_layout=True, figsize=(30, 6))

    for i, cm in enumerate(cm_list):
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 30}, cmap='Blues', ax=ax[i], cbar=False)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        heatmap.yaxis.set_ticks_position('left')
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, fontsize=20)
        heatmap.xaxis.set_ticks_position('top')

        ax[i].set_ylabel('Predicted Class', fontsize=20)
        ax[i].set_xlabel('Actual Class', fontsize=20)
        ax[i].set_title(f'Confusion Matrix - k = {i}', fontsize=24, pad=10)

        ax[i].xaxis.set_label_position('top')
        

def plot_learning_curve(estimator, title, X, y, cv=5,
                        n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title, fontsize=18)

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt