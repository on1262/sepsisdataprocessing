import numpy as np

def interp(fx:np.ndarray, fy:np.ndarray, x:np.ndarray):
    # fx, fy: (N,)
    # x: dim=1
    assert(fx.shape[0] == fy.shape[0] and len(fx.shape) == len(fy.shape) and len(fx.shape) == 1 and fx.shape[0] >= 1)
    assert(len(x.shape) == 1)
    if fx.shape[0] >= 2 and (not np.all(fx[:-1] < fx[1:])): # sort x, remove duplicated fx
        idx = sorted(list(range(fx.shape[0])), key=lambda x:fx[x])
        fx, fy = fx[idx], fy[idx]
        duplicate = np.zeros_like(fx, dtype=bool)
        duplicate[1:] = (fx[:-1] == fx[1:])
        if np.any(duplicate):
            fx = fx[duplicate == False]
            fy = fy[duplicate == False]
    elif fx.shape[0] == 1: # constant extrapolation
        return np.ones_like(x) * fy[0]
    
    result = np.ones_like(x) * fy[0]
    for idx, px in enumerate(fx):
        result[x >= px] = fy[idx] # very slow
    return result
        
if __name__ == '__main__':
    fx = np.asarray([1, 1, 1, 2])
    fy = np.asarray([-5, -30, 10, 20])
    x = np.asarray([1,2,0.1])
    print(interp(fx, fy, x))