from .processing import combine_and_select_samples, fix_feature_error_in_old_sys
import pandas as pd
from .config_loader import GLOBAL_CONF_LOADER


def scripts_fix_feature_error_in_static_data():
    old_csv = GLOBAL_CONF_LOADER['dataset_static']['paths']['csv_origin_path']
    new_csv = GLOBAL_CONF_LOADER['dataset_dynamic']['paths']['csv_raw_new_path']
    fix_feature_error_in_old_sys(old_csv=old_csv, new_csv=new_csv, output=old_csv)

def scripts_combine_and_select_samples():
    old_in = GLOBAL_CONF_LOADER['dataset_static']['paths']['csv_origin_path']
    new_in = GLOBAL_CONF_LOADER['dataset_dynamic']['paths']['csv_raw_new_path']
    data_a = pd.read_csv(old_in, encoding='utf-8')
    data_b = pd.read_csv(new_in, encoding='utf-8')
    data_combined = combine_and_select_samples(data_a, data_b, ['sta_', 'dyn_'])
    data_combined.to_csv(GLOBAL_CONF_LOADER['dataset_dynamic']['paths']['csv_origin_path'], encoding='utf-8',index=False)

