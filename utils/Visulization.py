from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import torch
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerLine2D
def updateline(handle, orig):
    handle.update_from(orig)
    handle.set_markersize(30)

totalColor = ['#e9546b',
              '#b68d4c',
              '#55295b',
              '#5b6356',
              '#706caa', 
              '#00a497', 
              '#c5c56a', 
              '#fef263', 
              '#16160e',
              '#89c3eb']
def visulization(dataname, inputX, labels, numberOfClass, method='pca', p=50, **kwargs):
    global totalColor
    colors = totalColor[:numberOfClass]
    labels = labels.cpu().numpy()
    print('number of class: '+str(numberOfClass))
    
    inputX = inputX - inputX.mean(0, keepdims=True)
    print(inputX)
    if method == 'pca':
        u, s, v = torch.pca_lowrank(inputX)
        output = torch.matmul(inputX, v[:, :2])
    elif method == 'mds':
        inputX = inputX.numpy()
        output = MDS(n_components=2).fit_transform(inputX)
    else:
        inputX = inputX.numpy()
        output = TSNE(n_components=2, perplexity=p, *kwargs).fit_transform(inputX)
    plt.cla()
    ax = plt.figure(figsize=(12, 12))
    colors = totalColor[:numberOfClass]
    for i in range(numberOfClass):
        plt.scatter(output[labels == i, 0], output[labels==i, 1], s=20, color=colors[i], label = 'Class '+str(i))
    # plt.legend(loc="upper left", fontsize=20)
    ax.legend(framealpha=1, handler_map={plt.Line2D: HandlerLine2D(update_func=updateline)},
              frameon=True, fontsize=28)
    plt.axis('off')
    plt.show()
    plt.savefig(dataname+'_visulization_'+method+'.jpg')
    

