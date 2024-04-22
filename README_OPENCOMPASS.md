---
- name：Datasets;
- desc：c-eval!!!
- language：-en
- dimension：examination
- sub_dimension:- sub_tag_1
- website：https://arxiv.org/abs/2305.08322
- github：https://github.com/hkust-nlp/ceval
- paper：https://arxiv.org/abs/2305.08322
- release_date:2023
- download_url：https://github.com/hkust-nlp/ceval
- cn: # optional, for chinese version website
    name: c-eval
    desc: c-eval中文数据集
---
## Introduction
This is a dataset for evaluating the performance of Chinese language models on various tasks.
## Meta Data
The data set has
- Question: The body of the question
- A, B, C, D: The options which the model should choose from
- Answer: (Only in dev and val set) The correct answer to the question
- Explanation: (Only in dev set) The reason for choosing the answer.
## Example
## Citation
```@article{huang2023ceval,
title={C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models}, 
    author={Huang, Yuzhen and Bai, Yuzhuo and Zhu, Zhihao and Zhang, Junlei and Zhang, Jinghan and Su, Tangjun and Liu, Junteng and Lv, Chuancheng and Zhang, Yikai and Lei, Jiayi and  Fu, Yao and Sun, Maosong and He, Junxian},
    journal={arXiv preprint arXiv:2305.08322},
    year={2023}
} ```