import tools
from tools.logging import logger
import matplotlib.pyplot as plt
from ..container import DataContainer
import numpy as np
from tqdm import tqdm
import os
from os.path import join as osjoin
import pandas as pd
import yaml
from datasets.derived_vent_dataset import MIMICIV_Vent_Dataset
from datasets.derived_ards_dataset import MIMICIV_ARDS_Dataset
from datasets.derived_raw_dataset import MIMICIV_Raw_Dataset

class DatasetReport():
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.paths = params['paths']
        if params['dataset_name'] == 'ards':
            self.dataset = MIMICIV_ARDS_Dataset()
        elif params['dataset_name'] == 'raw':
            self.dataset = MIMICIV_Raw_Dataset()
        elif params['dataset_name'] == 'vent':
            self.dataset = MIMICIV_Vent_Dataset()
        else:
            logger.error('Incorrect dataset_name')
            assert(0)
        self.dataset.load_version(params['dataset_version'])
        self.dataset.mode('all')
        self.gbl_conf = container._conf
        self.data = self.dataset.data
    
    def run(self):
        out_dir = os.path.join(self.paths['out_dir'], f'report_{self.params["dataset_name"]}')
        tools.reinit_dir(out_dir, build=True)
        report_path = osjoin(out_dir, f'dataset_report_{self.params["dataset_version"]}.txt')
        dist_dir = os.path.join(out_dir, 'dist')
        dir_names = ['points', 'duration', 'frequency', 'dynamic_value', 'static_value']
        for name in dir_names:
            os.makedirs(os.path.join(dist_dir, name), exist_ok=True)
        logger.info('generating dataset report')
        write_lines = []
        if self.params['basic']:
            # basic statistics
            write_lines.append('='*10 + 'basic' + '='*10)
            write_lines.append(f'Version: {self.params["dataset_version"]}')
            write_lines.append(f'Static keys: {len(self.dataset.static_keys)}')
            write_lines.append(f'Dynamic keys: {len(self.dataset.dynamic_keys)}')
            write_lines.append(f'Subjects:{len(self.dataset)}')
            write_lines.append(f'Static feature: {[self.dataset.fea_label(id) for id in self.dataset.static_keys]}')
            write_lines.append(f'Dynamic feature: {[self.dataset.fea_label(id) for id in self.dataset.dynamic_keys]}')
        if self.params['dynamic_dist']:
            # dynamic feature explore
            for id in tqdm(self.dataset.dynamic_keys, 'plot dynamic dist'):
                fea_name = self.dataset.fea_label(id)
                save_name = tools.remove_slash(str(fea_name))
                write_lines.append('='*10 + f'{fea_name}({id})' + '='*10)
                arr_points = []
                arr_duration = []
                arr_frequency = []
                arr_avg_value = []
                for s in self.dataset._subjects.values():
                    for adm in s.admissions:
                        if id in adm.keys():
                            dur = adm[id][-1,1] - adm[id][0,1]
                            arr_points.append(adm[id].shape[0])
                            arr_duration.append(dur)
                            if dur > 1e-3: # TODO 只有一个点无法计算
                                arr_frequency.append(arr_points[-1] / arr_duration[-1])
                            else:
                                arr_frequency.append(0)
                            arr_avg_value.append(adm[id][:,0].mean())
                arr_points, arr_duration, arr_frequency, arr_avg_value = \
                    np.asarray(arr_points), np.asarray(arr_duration), np.asarray(arr_frequency), np.asarray(arr_avg_value)
                if np.size(arr_points) != 0:
                    write_lines.append(f'average points per admission: {arr_points.mean():.3f}')
                if np.size(arr_duration) != 0:
                    write_lines.append(f'average duration(hour) per admission: {arr_duration.mean():.3f}')
                if np.size(arr_frequency) != 0:
                    write_lines.append(f'average frequency(point/hour) per admission: {arr_frequency.mean():.3f}')
                if np.size(arr_avg_value) != 0:
                    write_lines.append(f'average avg value per admission: {arr_avg_value.mean():.3f}')
                # plot distribution
                titles = ['points', 'duration', 'frequency', 'dynamic_value']
                arrs = [arr_points, arr_duration, arr_frequency, arr_avg_value]
                for title, arr in zip(titles, arrs):
                    if np.size(arr) != 0:
                        tools.plot_single_dist(
                            data=arr, data_name=f'{title}: {fea_name}', 
                            save_path=os.path.join(dist_dir, title, save_name + '.png'), discrete=False, adapt=True, bins=50)
        if self.params['static_dist']:
            # static feature explore
            for id in tqdm(self.dataset.static_keys, 'generate static feature report'):
                fea_name = self.dataset.fea_label(id)
                save_name = tools.remove_slash(str(fea_name))
                write_lines.append('='*10 + f'{fea_name}({id})' + '='*10)
                idx = self.dataset.idx_dict[str(id)]
                static_data = self.dataset.data[:, idx, 0]
                write_lines.append(f'mean: {static_data.mean():.3f}')
                write_lines.append(f'std: {static_data.std():.3f}')
                write_lines.append(f'max: {np.max(static_data):.3f}')
                write_lines.append(f'min: {np.min(static_data):.3f}')
                tools.plot_single_dist(
                    data=static_data, data_name=f'{fea_name}', 
                    save_path=os.path.join(dist_dir, 'static_value', save_name + '.png'), discrete=False, adapt=True, bins=50)
        # write report
        with open(report_path, 'w', encoding='utf-8') as fp:
            for line in write_lines:
                fp.write(line + '\n')
        logger.info(f'Report generated at {report_path}')