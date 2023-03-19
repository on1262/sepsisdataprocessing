import tools
import numpy as np

def test_plot_loss():
    data = {
        'train': np.random.normal(size=(5, 70)),
        'valid': np.random.normal(size=(5,70))
    }
    tools.plot_loss(data, std_bar=True, title='test title', out_path=None)
    data.pop('valid')
    tools.plot_loss(data, std_bar=True, title='test title', out_path=None)

if __name__ == '__main__':
    test_plot_loss()