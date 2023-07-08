# MIMIC-IV data processing pipeline

## 架构

该框架主要包括三个部分：数据集dataset、模型model、分析器analyzer。dataset将原数据抽象为torch.dataset接口；model对批次输入计算输出；analyzer类似trainer，提供K-fold、指标计算、绘图等工作。将model和analyzer拆分，使得一个analyzer调用多个model进行集成学习、一个model被多个analzyer调用等情况更加方便。其余的tools部分包括共用的工具方法，configs部分为需要配置的字段，例如路径、数据清洗的参数等。

analyzer: 分析模块
1. analyzer: 按照序列运行anlayzer，添加新analyzer时需要注册
2. container: 存放与模型无关的参数
3. feature_explore: 生成数据集的探查报告，可配置生成参数
3. utils: 工具方法

configs: 每个数据集对应的配置文件
1. global_config: 配置路径
2. config_manual: 手动配置, 能够覆盖自动生成的配置
3. config_cahce: 自动生成的配置文件, 不应被修改

data: 数据集文件

datasets: 数据集抽象, 包括数据提取/清洗/重新组织

libs: 第三方库和相关代码

models: 模型

outputs: 输出图片/分析表等

tools: 工具类

main.py: 主入口, 通过参数配置和luncher对接

## 数据集

micmic-iv: MIMIC-IV数据集

框架本身可以支持多个数据集，且对一个数据集可以产生多个“版本”（version），不同版本的数据集可以有不同的特征数量和处理方法，便于特征筛选和不同数据集的对照设计

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

## 特征加工

PF_ratio: PaO2/FiO2

MAP(平均动脉压): (SBP(收缩压) + 2*DBP(舒张压))/3

shock_index(休克指数): HR(心率) / SBP(收缩压)

PPD(pulse pressure difference,脉压差): SBP - DBP

