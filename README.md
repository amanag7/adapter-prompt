Adapter-Prompt
==============================

Using adapters along with smaller LMs to perform better prompt tuning


<!-- <p align="center">
<img src=""  />
</p>


Project Organization
------------
```
├── api
│   ├── app.py
│   ├── config
│   │   └── config.py
│   ├── resources
│   ├── static
│   └── templates
│   
├── checkpoints
│   
├── data
│   ├── processed
│   └── raw
│   
├── docs
│   ├── Analysis.md
│   └── Requirements.md
│   
├── notebooks
│ 
├── Adapter-Prompt
│   ├── config
│   │   └── config.py
│   ├── data
│   │   └── make_dataset.py
│   ├── dispatcher
│   ├── features
│   │   ├── build_features.py
│   ├── models
│   │   ├── test_model.py
│   │   └── train_model.py
│   ├── utils
│   ├── visualisation
│   |   └── visualisation.py
│   └── main.py
│ 
├── Dockerfile
│ 
├── run.sh
├── logs
├── references
├── requirements.txt
├── README.md
├── LICENSE
└── tests
    └── test_environment.py
```
-------- -->


## Getting Started

### Setup requirements

```
poetry install
```

## Dataset

| Task | Dataset (Original Data Link) |
| ---- | ------- |
| Question Answering | [SQuAD version 1.1](https://rajpurkar.github.io/SQuAD-explorer/) |
| Machine Translation | [IWSLT](https://wit3.fbk.eu/mt.php?release=2016-01) |
| Summarization | [CNN/DM](https://cs.nyu.edu/~kcho/DMQA/) |
| Natural Language Inference | [CNN/DM](https://www.nyu.edu/projects/bowman/multinli/) |
| Sentiment Analysis  | [SST](https://nlp.stanford.edu/sentiment/treebank.html) |
| Semantic Role Labeling | [QA‑SRL](https://dada.cs.washington.edu/qasrl/) |
| Zero-Shot Relation Extraction | [QA‑ZRE](http://nlp.cs.washington.edu/zeroshot/) |
| Goal-Oriented Dialogue | [WOZ](https://github.com/nmrksic/neural-belief-tracker/tree/master/data/woz) |
| Semantic Parsing | [WikiSQL](https://github.com/salesforce/WikiSQL) |
| Commonsense Reasoning | [MWSC](https://s3.amazonaws.com/research.metamind.io/decaNLP/data/schema.txt) |
| Text Classification | [AGNews, Yelp, Amazon, DBPedia, Yahoo](http://goo.gl/JyCnZq) |

In order to unify the format of all the dataset, we first ran the code in https://github.com/salesforce/decaNLP to get the first 10 transformed dataset, and then converted them into Squad-like format. For the last 5 dataset, we converted them directly. All converted dataset are available [here](https://drive.google.com/file/d/1rWcgnVcNpwxmBI3c5ovNx-E8XKOEL77S/view?usp=sharing).


<!-- ### Run

```
python -m Adapter-Prompt.main
```
OR

```
./run.sh
```

### Test

```
python -m tests.test_environment
```


### To-do List

- [ ] Download dataset
- [ ] Pre-process data
- [ ] Train model
- [ ] Test model
- [ ] Main Pipeline

------------------------------- -->
