# Sepsis induced ARDS

## 架构

analyzer: 分析模块
1. analyzer: 包含与各种model的接口
2. analyzer_utils: 工具方法

configs: 每个数据集对应的配置文件
1. global_config: 配置路径
2. config_manual: 手动配置, 能够覆盖自动生成的配置
3. config_cahce: 自动生成的配置文件, 不应被修改

data: 数据集文件

datasets: 数据集抽象, 包括数据提取/清洗/重新组织
1. abstract_dataset: 所有数据集需要实现的接口

libs: 第三方库和相关代码

models: 模型, 包括baseline/LSTM等

outputs: 输出图片/分析表等

tools: 工具类

main.py: 主入口, 通过参数配置和luncher对接

## 数据集

注意: 数据集的区分是因为特征数量/采样时间地点不同而产生显著差异.

micmic-iv: MIMIC-IV数据集

NanjingA: 东南大学附属中大医院的数据集

JiangsuA(采集中):来自江苏省内多家医院的数据集 

## sepsis3.csv 生成

**build postgresql**

createdb mimiciv

cd ~/mimic-code/mimic-iv/buildmimic/postgres

psql -d mimiciv -f create.sql

psql -d mimiciv -v ON_ERROR_STOP=1 -v mimic_data_dir=/home/chenyt/sepsisinducedards/mimic-iv -f load.sql

**build concepts**

cd ~/mimic-code/mimic-iv/concepts_postgres

psql -d mimiciv

\i postgres-functions.sql -- only needs to be run once

\i postgres-make-concepts.sql
 
**extract csv**

\copy (SELECT * FROM mimiciv_derived.sepsis3) TO '~/sepsisinducedards/mimic-iv/cache/sepsis3.csv' WITH CSV HEADER; -- 提取出csv文件

## feature engineering

PF_ratio: PaO2/FiO2

MAP(平均动脉压): (SBP(收缩压) + 2*DBP(舒张压))/3

shock_index(休克指数): HR(心率) / SBP(收缩压)

PPD(pulse pressure difference,脉压差): SBP - DBP

