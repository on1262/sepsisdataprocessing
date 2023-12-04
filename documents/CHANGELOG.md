# CHANGELOG

## 2023.8.18

- Fixed the missing cache folder when mimic_dataset is first preprocessed, or you can fix it manually by creating a new `cache` empty folder under `data/mimic-iv`.
- Added instructions for configuring the pytorch CUDA version in the readme

## 2023.10.26

Upgrade to generic framework

### Important changes
- Improved code readability
- Decoupled generic code and Sepsis/ARDS-specific processing code, customized processing by modifying abstract functions through derived classes.
- Improved data processing speed, it takes about 40 minutes to process the whole MIMIC-IV dataset.
- Include new data such as hosp.labevents/ed.vitalsign, linking MIMIC-IV and ED datasets.
- Fixed potential data leakage by changing linear interpolation to historical nearest neighbor padding
- 

### Other changes
- Compressed storage cache files and read only necessary files.
- Iterative culling of high missing samples and high missing features, this algorithm gives better results
- Configuration files are now in yaml format for easier annotation.

## 2023.11.24

### Important changes
- remove everything about sepsis in mimic-iv-raw dataset
- add example pipeline: ventilation
- mimic_dataset->mimic_ards_dataset. All these 3 datasets are examples
- New ways of interpolation
- nearest_static will be changed to latest_static. It will only obtain latest history. return -1 if no history is founded
- empty align target: push forward the start time when at least one dynamic feature is valid

### Other changes
- fix bug: admittime < dischtime
- fix bug: carrunit now is in static data
- fix bug: generate label
- empty align id is available

## 2023.11.30

- compressing format default: xz -> gz (faster)
- value->valuenum in hosp extraction
- add category filter (>2%)
- 