# CHANGELOG

## 23.8.18

- 修复了mimic_dataset第一次预处理时缺失cache文件夹的问题，也可以手动修复，在`data/mimic-iv`下新建`cache`空文件夹即可
- 在readme中增加了配置pytorch CUDA版本的指导

## 23.10.26

升级为通用框架

### 重要改动
- 提高了代码可读性
- 解耦通用代码和Sepsis/ARDS特异性处理的代码，通过派生类修改抽象函数实现自定义处理
- 提升数据处理速度，处理整个MIMIC-IV数据集需要40分钟左右
- 囊括了hosp.labevents/ed.vitalsign等新数据，链接了MIMIC-IV和ED数据集
- 将线性插值改为历史最近邻填充，修复了潜在的数据泄漏问题
- 

### 其他改动
- 压缩存储cache文件，并且只读取必要的文件。
- 迭代剔除高缺失样本和高缺失特征，这种算法能得到更理想的结果
- 配置文件改用yaml格式，便于注释