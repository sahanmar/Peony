# Peony Project

## Active Learning for Text Classification

![](https://github.com/sahanmar/Peony/blob/supporting_images/images/peony.png)

### Abstract 

Following project is aimed on derivation and testing of active learning techniques for text classification. Active learning method is introduced as semi-supervised learning algorithm that uses annotator's help. When an algorithm is not sure about a label of a specific instance, it will ask an annotator to provide the label. On the basis of the iterative annotators feedback the algorithms have more powerful ability for learning with lower amount of training documents. In this work, term classification is interpreted in different ways such as named entity recognition, anomaly context detection, binary classification, multi-class classification, etc.. In order to solve highlighted problems, in this work are used different methods such as random forests, svms, neural networks, etc.. Each method is defined with respect to decision theory paradigm and tested on real data with visualised results.

### Repository Guide

This Project is separated into two folders *Peony_research_document* and *Peony_project*. 

*Peony_research_document* is a folder with output PDF document and .tex, .lyx files.

*Peopny_project* is a tech folder with codes.

### Peony Deployment

In order to start working with all Peony dependecies `peony_project` environment must be activated. The easiest way to activate peony environmet is to run  `conda env create -f environment.yml` (If you don't have conda, please install it before creating environment). 
