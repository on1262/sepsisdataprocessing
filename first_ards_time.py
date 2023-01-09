import pandas as pd
import numpy as np
import tools

# 计算第一次ards时间的分布
def first_ards_time_dist(csv_path):
    df = pd.read_csv(csv_path)
    d_dict = {
        'D1':1,
        'D2':2,
        'D3':3,
        'D4':4,
        'D5':5,
        'D6':6,
        'D7':7
    }
    print(sorted(df.columns))
    col_dict = {}
    for col in df.columns:
        for key in d_dict.keys():
            if key in col:
                col_dict[col] = d_dict[key]
    
    times = []
    for r_idx, row in df.iterrows():
        for col in sorted(df.columns):
            if row[col] <= 300:
                times.append(col_dict[col])
                break
        times.append(-1)

    times = np.asarray(times)
    tools.plot_single_dist(times, 'first_ards_time')
    
                
                
if __name__ == '__main__':
    csv_path = r'F:\\Project\DiplomaProj\\new_data\\7d_pao2_fio2.csv'
    first_ards_time_dist(csv_path)
