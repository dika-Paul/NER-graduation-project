# 面向材料科学文献的命名实体识别项目

本仓库是毕业设计项目源码，主要实现面向材料科学英文文献的命名实体识别（Named Entity Recognition, NER）与大模型辅助伪标注迭代训练流程。项目目标是从论文标题、摘要等文本中自动抽取材料名称、材料性能、表征方法、合成方法、应用场景等领域实体，为材料文献结构化分析和知识图谱构建提供基础。

## 项目功能

- 支持 MatScholar 数据集转换为 BIO 序列标注格式。
- 实现多种 NER 模型，包括 BiLSTM-CRF、BERT-Softmax、BERT-BiLSTM-CRF 和 MatSciBERT-Softmax。
- 使用 precision、recall、F1 和 loss 对模型效果进行统一评估。
- 基于 LangGraph 构建大模型辅助伪标注流程，支持模型预测、LLM 抽取、差异度计算、结果筛选和训练集追加。
- 支持从 OpenAlex Excel 文献池批量读取论文标题和摘要，并转换为句子级样本参与迭代训练。

## 实体类型

项目使用材料科学领域常见实体标签：

| 标签 | 含义 |
| --- | --- |
| MAT | 材料、化合物、复合材料、掺杂元素等 |
| SPL | 晶相、结构相、对称性标签等 |
| DSC | 材料描述、形貌、结构、样品形态等 |
| PRO | 材料性质、性能指标等 |
| APL | 应用场景、器件、功能目标等 |
| SMT | 合成方法、制备路线、加工工艺等 |
| CMT | 表征方法、测试技术、实验仪器等 |

## 项目结构

```text
NRE_project/
├── README.md                    # GitHub 仓库说明
├── evaluate.py                  # 模型评估函数
├── data/                        # 训练与复现实验数据
├── docs/                        # 毕业设计提交说明文档
├── models/                      # NER 模型结构定义
├── utils/                       # 数据读取、转换与 collate 工具
├── train/                       # 模型训练 Notebook
├── test/                        # 模型测试 Notebook
└── graph/                       # 大模型辅助伪标注迭代流程
```

## 核心方法

项目首先使用人工标注数据训练基础 NER 模型，然后对未标注文献进行模型预测和大模型实体抽取。系统将 NER 输出与大模型输出转换为统一实体字典，并通过编辑距离差异比例判断结果一致性。当预测结果置信度较高时，系统将样本转换为 BIO 格式追加到训练集中，从而实现伪标注样本扩充和迭代训练。

## 运行说明

1. 安装 Python 环境，并准备 PyTorch、Transformers、torchcrf、seqeval、rapidfuzz、LangChain、LangGraph 等依赖。
2. 数据文件已放在 `data/` 目录，模型权重和运行输出默认不提交到仓库。
3. 如需重新生成 MatScholar BIO 数据，可运行 `utils/convert_matscholar_to_bio.py`。
4. 单模型训练可打开 `train/` 目录下对应 Notebook。
5. 大模型辅助迭代训练可使用 `graph/graph.py` 中的 `build_train_graph()` 或 `build_add_train_graph()` 构建流程。

## 毕业设计附件

仓库已提供两份 Markdown 说明文档，可用于毕业设计提交附件：

- `docs/项目简略版.md`
- `docs/毕业设计提交附件.md`

## 项目成果

本项目完成了材料领域命名实体识别的数据处理、模型训练、模型评估和大模型辅助伪标注流程设计。项目可以作为材料科学文献自动信息抽取的实验基础，也可扩展用于材料知识图谱构建、文献智能检索和材料数据挖掘等后续任务。
