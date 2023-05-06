import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    x = np.arange(0, 16, 1)
    legend = []
    for dp in np.linspace(0, 1, 11):
        y = np.zeros(x.shape)
        for idx in range(1, len(y)+1):
            y[idx-1] = math.comb(16, idx)*(dp**idx)*((1-dp)**(16-idx))
        plt.plot(x, y[:])
        legend.append(f'dp={dp:.2f}')
    plt.legend(legend)
    plt.savefig('eval.png')
    plt.close()
