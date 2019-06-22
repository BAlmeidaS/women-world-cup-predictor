import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def print_confusion_matrix(confusion_matrix, labels, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=labels, columns=labels, 
    )

    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 16}, cmap='Blues')
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