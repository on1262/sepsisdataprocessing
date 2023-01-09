import processing
import os


def scripts_fix_feature_error_in_old_sys():
    old_csv = r"F:\Project\DiplomaProj\data\sepsis.csv"
    combined_csv = r"F:\Project\DiplomaProj\new_data\data_combined.csv"
    output_csv = r"F:\Project\DiplomaProj\data\sepsis_fixed.csv"
    processing.fix_feature_error_in_old_sys(old_csv, combined_csv, output_csv)

if __name__ == "__main__":
    scripts_fix_feature_error_in_old_sys()
