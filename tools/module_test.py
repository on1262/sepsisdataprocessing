import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm:np.ndarray, labels:list, title='Confusion matrix', save_path='./out.png'):
    '''
    生成混淆矩阵
    cm: 沿axis=0是predicted label轴, 沿axis=1是true label轴, cm[x][y]代表pred=x, gt=y
    labels: list(str) 各个class的名字
    save_path: 完整路径名
    '''
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 10))
    plt.gca().grid(False)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.OrRd)
    plt.title(title, size=18)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, size=15)
    plt.yticks(tick_marks, labels, size=15)
    plt.ylabel('True label', size=18)
    plt.xlabel('Predicted label', size=18)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            num_color = 'black' if cm[x][y] < 1.5*cm.mean() else 'white'
            plt.annotate(str(cm[x][y]), xy=(y, x), fontsize=24, color=num_color,
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    # Example usage
    labels = ['class 0', 'class 1', 'class 2', 'class 3']
    cm = np.asarray([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,1,0,0]])
    plot_confusion_matrix(cm, labels, title='Confusion matrix')
    