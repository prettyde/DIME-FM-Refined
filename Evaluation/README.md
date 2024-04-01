
# DIME-FM Refined Implementation

## About

This repository is a comprehensive guide and toolkit to setup, understand, and execute a refined implementation of the DIstilling Multimodal and Efficient Foundation Models - DIME-FM. Documentation includes original repository contents now improved for better understanding under new management by prettyde.

## Setting up the Evaluation Environment
Follow instructions similar to the original ELEVATER toolkit README. This refined version contains fewer ambiguous terminologies and clearer instructions.

### Download Datasets
The datasets originally not included in the benchmark can be downloaded from the provided links. These datasets are used for the Robustness Evaluation of the Vision-Language Model.

Each entry in the table below includes a dataset and respective download link.

|     Dataset      | Download Link | 
|:----------------:|:-------------:|
|ImageNet-v2|[dataset](https://drive.google.com/file/d/1u14z5pI8lsbEM6XUNvFWNboFQ11mCVuM/view?usp=sharing)|
|ImageNet-A|[dataset](https://drive.google.com/file/d/1WNth46yXZ5l8W2jBl50nu4p5bmv2qUil/view?usp=drive_link)|
|ImageNet-Sketch|[dataset](https://drive.google.com/file/d/1_V5gJtwIy1-iyXSyxFKlfundtZ_S6q1M/view?usp=sharing)|
|ImageNet-R|[dataset](https://drive.google.com/file/d/1f13nKHOLgjoW6B9V1cqWMcfaLt06xUJv/view?usp=drive_link)|
|ObjectNet(IN-1K)|[dataset](https://drive.google.com/file/d/11wKLGJyF0BMdgWzFAwj4hnXn4jF8Fk2g/view?usp=sharing)|

## Usage

After downloading and appropriately setting up the datasets, proceed to use the evaluation commands as found in the original repository.

Continue using the linear probing and zero-shot evaluation commands for both Transferability and Robustness.


# Documents and Resources
This section includes resources that you may find essential in understanding various concepts, definitions, and terms:
- ELEVATER Image Classification Toolkit
- Installation guide
- Dataset documentation
- Getting started guide
- Evaluation on various fronts

## Citation
Please refer to and cite the original document whenever you make use of these resources:

```bibtex
@article{li2022elevater,
    title={ELEVATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models},
    author={Li, Chunyuan and Liu, Haotian and Li, Liunian Harold and Zhang, Pengchuan and Aneja, Jyoti and Yang, Jianwei and Jin, Ping and Lee, Yong Jae and Hu, Houdong and Liu, Zicheng and Gao, Jianfeng},
    journal={Neural Information Processing Systems},
    year={2022}
}
```