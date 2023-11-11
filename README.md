
![Pipeline Overview](document/general_pipeline.png)

# MIMIC-IV Data Processing Pipeline

<center> [中文版本](README_CN.md) | [English Version](README.md) </center>

## Structure

The framework consists of three main parts: dataset dataset, model model, and analyzer analyzer. dataset abstracts the original data into torch.dataset interface; model computes outputs on batch inputs; and analyzer is similar to trainer to provide K-fold, metrics computation, plotting, and other tasks. Splitting model and analyzer makes it more convenient for a single analyzer to call multiple models for integrated learning, and for a single model to be called by multiple analzyers. The rest of the tools section includes common tool methods, and the configs section is for fields that need to be configured, such as paths, parameters for data cleansing, and so on.

analyzer: trainer/visualization scripts/...
1. analyzer: runs the analyzer sequentially and needs to be registered when a new analyzer is added.
2. container: store model-independent parameters
3. feature_explore: generates an exploratory report of the dataset, with configurable generation parameters
4. utils: utility methods

configs: Configuration files for each dataset.
1. global_config: Configuration path
2. config_manual: Manual configuration, can override the automatically generated configuration.
3. config_cache: Automatically generated configuration file, should not be modified.

Other modules:
- data: dataset files
- datasets: dataset abstraction, including data extraction/cleaning/reorganisation
- libs: third-party libraries and related code
- models: models
- outputs: output images/analysis tables, etc.
- tools: tool classes
- main.py: main entry, interfaces with the launcher via parameter configuration.

## Deployment


按照以下步骤部署：
1. 在`python=3.10.11`环境下配置conda环境，并安装所需的packages：`pip install -r requirements.txt`
2. 第一步中关于pytorch的cuda版本问题参考下一小节
3. 将MIMIC-IV数据集解压至`data/mimic-iv`文件夹下, 子文件夹有`hosp`,`icu`等
4. 将生成的`sepsis3.csv`存放在`data/mimic-iv/sepsis_result`下
5. 运行`python -u main.py`，生成整个数据集需要一个半小时左右，第一次运行会生成数据集探查结果和一个实例模型的预测结果

安装Pytorch对应的CUDA版本：
1. 新建并进入一个conda虚拟环境
2. 输入`nvidia-smi` 查看服务器安装的CUDA版本
3. 按照 https://pytorch.org/get-started/previous-versions/ 选择linux下的对应版本的安装命令，pytorch对应的CUDA版本可以落后于服务器的CUDA版本
4. 检查是否安装成功： https://blog.csdn.net/qq_45032341/article/details/105196680
5. 如果安装了不同于`requirements.txt`中的pytorch版本，将对应的行删掉，避免重复安装
6. 这个框架本身对第三方库的版本没有严格限制

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

提取的sepsis3表格有**32971**个行，**25596**个患者，包含的列有：
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

```
createdb mimiciv

cd ~/mimic-code/mimic-iv/buildmimic/postgres

psql -d mimiciv -f create.sql

psql -d mimiciv -v ON_ERROR_STOP=1 -v mimic_data_dir=/path/to/mimic-iv -f load.sql

```
**step2: build concepts**

```
cd ~/mimic-code/mimic-iv/concepts_postgres

psql -d mimiciv

\i postgres-functions.sql -- only needs to be run once

\i postgres-make-concepts.sql
 
```
**step3: extract csv**

```

\copy (SELECT * FROM mimiciv_derived.sepsis3) TO '~/sepsis3.csv' WITH CSV HEADER; -- 提取出csv文件

```

## 特征加工 & 新增特征

新增特征：
- PF_ratio: PaO2/FiO2
- MAP(平均动脉压): (SBP(收缩压) + 2*DBP(舒张压))/3
- shock_index(休克指数): HR(心率) / SBP(收缩压)
- PPD(pulse pressure difference,脉压差): SBP - DBP
- sepsis time: 相对于起始时刻的sepsis发生时间（小时）

data source: 样本对应的admission的来源位置
- icu: 样本对应的stay id在ICU stays内
- ed: 样本对应的transfer id在Emergency Department内

## dataset report & feature explorer

阐明一些特殊概念：
- config/miss_dict：新增特征计算缺失值时需要指定参与计算的特征，取输入特征缺失情况的交集
- global miss rate: 不受remove rule version2影响，可以看作所有sepsis患者的缺失情况，以subject为最小单位，覆盖static+dynamic feature
- hit table：按照remove rule去除不合适的患者后，剩下群体中的特征覆盖率（覆盖率=1-缺失率），以admission为最小单位，并且覆盖dynamic feature+additional feature
    - 预处理阶段会按照hit table去除覆盖率较小的特征，阈值为`remove_rule/min_cover_rate`
- miss mat: 去除不合适的患者后，剩余群体中的特征缺失情况分布