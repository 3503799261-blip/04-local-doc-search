# 本地 Markdown 文档检索助手

读取本地 Markdown，按标题和滑动窗口分块，再用字符 TF-IDF 检索原文片段，返回文件名、块编号和匹配分数。项目展示文档检索流程；它不调用大模型，不生成答案，也没有语义向量模型。

## 运行

在本项目目录使用 Python 3.11：

```bash
python -m pip install -r requirements.txt
python main.py
python main.py --query "图书快到期了可以续借吗"
python main.py --docs-dir ./data/docs --query "校园卡丢了怎么办" --top-k 2
python main.py --docs-dir /path/to/your/markdown --query "你的问题" --output-dir outputs/custom
```

Windows 自定义路径含空格时加引号。文档需要 UTF-8 编码，扩展名为 `.md`；脚本递归读取指定目录。自定义数据的评估会显示 `skipped`，不会套用示例语料的成绩。

## 数据、分块与评估

- `data/docs/` 是六份自编中文校园服务手册，所有学校规则、系统名和时间均为虚构教学内容。
- `data/queries.json` 是 24 条人工查询和相关来源文件名，不加入索引、不调节参数。
- 先按 Markdown 标题切分，文档标题与前言并入第一个正文节，避免短标题单独抢占排序。长段落使用 240 字窗口、40 字重叠；`--max-chars` 和 `--overlap` 可调整。原文片段保持不改写。
- TF-IDF 使用字符 2–3 gram，余弦分数排序。对照基线是查询与块的双字片段交集数量，不使用 IDF。
- 评估先对排序结果按来源文件去重，再计算 **来源级** MRR 与 Recall@3，避免同一文件的多个块挤占评估名额。它不是精确答案跨度评估，也不衡量生成质量。

零分片段不返回。非零词面匹配不能保证问题可回答；没有共享字词的同义问题可能检索失败，小数据不能代表真实知识库表现。

## 输出

默认写入脚本旁的 `outputs/`：

| 文件 | 内容 |
|---|---|
| `metrics.json` | 数据来源、分块配置、实际指标或跳过原因 |
| `evaluation.json` | 每个标注查询的相关来源与排序 |
| `retrieval_report.png` | 示例语料指标对比；自定义语料改为块长分布 |
| `index.json` | 分块后的原文及文件来源 |
| `demo.json` | 当前查询的原文候选、来源与分数 |

## 面试可以讲什么

分块长度与重叠如何影响召回、为什么不能把词面分数当正确率、块级与文档级评估的差别，以及检索和生成分别负责什么。下一步可替换为语义向量检索，增加人工标注的相关块，再考虑附引用的生成式回答。

参考：[scikit-learn 文本特征提取](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)。代码与自编数据采用本目录 MIT 许可。

## 实际示例结果

见[本次实际运行结果](RESULTS.md)，包含生成的指标和结果图。重新运行 `python main.py` 可以复现流程。
