# Multimedia Generative Script Learning for Task Planning

[Multimedia Generative Script Learning for Task Planning](https://arxiv.org/abs/2208.12306)

Accepted by Findings of the Association for Computational Linguistics: ACL 2023

Table of Contents

=================

* [Overview](#overview)
  
* [Requirements](#requirements)
  
* [Quickstart](#quickstart)
  
* [Citation](#citation)

## Overview

<p align="center">
  <img src="https://eaglew.github.io/images/wikihow.png?raw=true" alt="Photo" style="width: 50%;"/>
</p>

## Requirements

### Environment

* Python 3.8.5
  
* Ubuntu 20.04

### Setup Instructions

To set up the environment for this repository, please follow the steps below:

Step 1: Create a Python environment (optional)
If you wish to use a specific Python environment, you can create one using the following:

```bash
conda create -n pyt1.11 python=3.8.5
```

Step 2: Install PyTorch with CUDA (optional)
If you want to use PyTorch with CUDA support, you can install it using the following:

```bash
conda install pytorch==1.11 torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Step 3: Install Python dependencies
To install the required Python dependencies, run the following command:

```bash
pip install -r requirements.txt
```

#### Data Description:

[Multimedia Goal-oriented Generative Script Learning Dataset](https://drive.google.com/file/d/1lSo-Kr4edNas0_uTl1SvDnEGuPYl0Or9/view?usp=sharing) 
This link contains a dataset consisting of multimedia steps for two categories: gardening and crafts. The dataset consists of a total of 79,089 multimedia steps across 5,652 tasks.

The dataset is split into three sets: training, development, and testing. The gardening category has 20,258 training tasks, 2,428 development tasks, and 2,684 testing tasks. The crafts category has 32,082 training tasks, 4,064 development tasks, and 3,937 testing tasks. Each task is associated with a set of multimedia steps, which include corresponding step images related to the task. 

Note that this dataset is specific to the gardening and crafts categories and may not be representative of other categories or domains.

The `*_data` folder contains the full dataset, which will be released after the paper is published. Each `*_data` folder includes three files: `train.json`, `valid.json`, and `test.json`. These files are used for training, validation, and testing respectively.

Each file is a JSON file that contains multiple lines. Each line represents an instance and follows the schema described below:

```python
{
    "title":        #   goal of activity
    "method":       #   subgoal of activity
    "steps":        #   list of step text 
    "captions":     #   list of corresponding captions of step
    "target":       #   next step text
    "img":          #   last step image id
    "target_img":   #   next step image id
    "retrieve":     #   20 retrieved historical relevant steps
    "retrieve_neg": #   list of retrieved top-20 most similar steps with respect to the last step. They will serve as retrieve-negatives 
    }
```

The `img` subfolder in the `*_data` folder contains all images and the corresponding wikihow task json file for the gardening and crafts datasets.

## Quickstart

Download and unzip the `Wiki_dataset.zip` from [Multimedia Goal-oriented Generative Script Learning Dataset](https://drive.google.com/file/d/1lSo-Kr4edNas0_uTl1SvDnEGuPYl0Or9/view?usp=sharing). Put `crafts_data` and `gardening_data` under the main folder.

### Training

To train the model on the crafts domain, run the following command:

```bash
bash finetune_crafts.sh 
```

To train the model on the gardening domain, run the following command:

```bash
bash finetune_garden.sh 
```
These scripts will run the training process and save the trained model to specified folders.

#### Test

To test the model on the crafts domain, run the following command:

```bash
bash test_crafts.sh 
```

To test the model on the gardening domain, run the following command:

```bash
bash test_garden.sh 
```

These scripts will load the saved trained model and run the testing process. 

Note that the training and testing scripts assume that the data is located in the default directory. If you have stored the data in a different directory, you will need to modify the scripts accordingly.

## Citation

```bib
@inproceedings{wang2022multimedia,
    title={Multimedia Generative Script Learning for Task Planning},
    author={Wang, Qingyun and Li, Manling and Chan, Hou Pong and Huang, Lifu and Hockenmaier, Julia and Chowdhary, Girish and Ji, Heng},
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = july,
    year = "2023",
    publisher = "Association for Computational Linguistics",
    }
```