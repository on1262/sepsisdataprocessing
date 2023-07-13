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
3. config_cache: 自动生成的配置文件, 不应被修改

data: 数据集文件

datasets: 数据集抽象, 包括数据提取/清洗/重新组织

libs: 第三方库和相关代码

models: 模型

outputs: 输出图片/分析表等

tools: 工具类

main.py: 主入口, 通过参数配置和launcher对接

## MIMIC-IV数据集

框架本身可以支持多个数据集，且对一个数据集可以产生多个“版本”（version），不同版本的数据集可以有不同的特征数量和处理方法，便于特征筛选和不同数据集的对照设计

**subjects、admissions和stays**

在hosp/transfer记录中可以看到一个subjects往往对应多个admissions，而且在Emergency Department的情况较多，还有病人在手术室的情况。

在icu/icu_stays中，admission和stay的比值为0.905:1，两者几乎是一对一的关系。相比hosp/transfer表，icu_stays记录的只是在ICU内的一部分，并且合并了连续的病房转移。即一个stay id对应一个或多个transfer id，但存在许多transfer id在stay id的范围外，包括急诊室、手术室的情况。同样地，并非所有admission都包括ICU内的情况，不是所有admission都拥有stay id

在数据清洗过程中，只抽取ICU和Emergency Department两个部分的内容，所有stays/transfers都被看作独立的admissions。一个患者存在多段住院经历，将会被看作多个样本。

**mimic_dataset/MIMICIV**

对MIMIC数据集进行比较底层的处理，包含三个phase：

phase1: 抽取sepsis患者ID和指标与编号的映射文件
- 载入sepsis3筛选结果（见sepsis3.csv生成）
- 载入hosptial lab item和ICU lab item编号表
- sepsis发生时间定义为 min(antibiotic_time, culture_time)

phase2: 抽取subject列表，补充患者的基本信息
- 按照sepsis3 result，抽取patients中的患者，构建subject dict，并添加基本信息
- 从icu_stays抽取ICU内患者信息，构建admissions
- 从transfers中抽取ED患者信息，构建admissions
- 从omr中抽取患者的静态指标，补充到subject中

phase3: 读取icu_events，记录动态特征
- 配置需要采集的指标，只有数值型的指标会被提取
- 提取icu_events表
- 清理不符合标准的admission和subject

**mimic_dataset/MIMICIV_Dataset**

MIMIC-IV数据集的上层抽象，转化为一个dim=3的矩阵

preprocess to numeric: 将载入信息化为数值型
- 类别特征到数值映射到转化（目前只做了gender）
- 将不规范存储到特征转化为数值型（舒张压、收缩压）
- 约束特征上下界取值

preprocess norm: 获取归一化信息
- 为每个特征计算均值和方差

preprocess table: 制作最终的数据集矩阵
- 布局：[static feature, dynamic feature, additional feature]
- 序列的起始时间是max(sepsis time, admittime)
- 特征工程新增特征
- 线性插值，没有数据的列或者时刻填充-1

preprocess version: 生成数据集的不同版本
- feature_limit: 数据集的特征只能在给定的范围中，空表示没有限制
- forbidden feature: 不能包含给定的特征，可以用于高相关度剔除
- data source: 选择数据集中包含的样本来自于ICU还是ED
- 生成K-Fold样本划分

## sepsis3.csv 生成

提取的sepsis3表格有**32971**个样本，包含的列有：
- subject_id
- stay_id
- antibiotic_time
- culture_time
- suspected_infection_time
- sofa_time
- sofa_score
- respiration
- coagulation
- liver
- cardiovascular
- cns
- renal
- sepsis3

**step1: build postgresql**

createdb mimiciv

cd ~/mimic-code/mimic-iv/buildmimic/postgres

psql -d mimiciv -f create.sql

psql -d mimiciv -v ON_ERROR_STOP=1 -v mimic_data_dir=/path/to/mimic-iv -f load.sql

**step2: build concepts**

cd ~/mimic-code/mimic-iv/concepts_postgres

psql -d mimiciv

\i postgres-functions.sql -- only needs to be run once

\i postgres-make-concepts.sql
 
**step3: extract csv**

\copy (SELECT * FROM mimiciv_derived.sepsis3) TO '~/sepsis3.csv' WITH CSV HEADER; -- 提取出csv文件

## 特征加工 & 新增特征

PF_ratio: PaO2/FiO2

MAP(平均动脉压): (SBP(收缩压) + 2*DBP(舒张压))/3

shock_index(休克指数): HR(心率) / SBP(收缩压)

PPD(pulse pressure difference,脉压差): SBP - DBP

sepsis time: 相对于起始时刻的sepsis发生时间（小时）

data source: 样本对应的admission的来源位置
- icu: 样本对应的stay id在ICU stays内
- ed: 样本对应的transfer id在Emergency Department内

