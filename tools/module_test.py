import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    a = np.random.randn(2,3,4)
    b = np.reshape(a, (6,4))
    print(a)
    print('='*20)
    print(b)