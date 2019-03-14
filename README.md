# Intro
This repository contains a naive implementation of Decision Trees and Random Forests. The variant this repository contains is called Classification and Regression Trees.

# Requirements
The following method works on Ubuntu 18.04 LTS. You need to install anaconda and start the cli and type in the following command to replicate the environment on your local machine.
```bash
conda create --name ENV_NAME --file Requirements.txt
```
Replace `ENV_NAME` with an environment name of your choice.

# Usage
* See `usage/decision-tree/assignment1.ipynb` for a commented usage of the implemented CART algo.
* See `usage/random-forest/assignment2.ipynb` for a commented usage of the implemented random forest.
* See `usage/knn/knn.ipynb` for a commented usage of the implemented K-Nearest Neighbour Algorithm for Classification and Regression tasks.
* Folder ``scripts`` contains python scripts to download the data used in the jupyter notebooks.
* Look at ``answer_sheet#.md`` for the assignment answers.

# Credits
The implementation I have here, takes ideas from Josh Gordan's implementation of Classification Decision Trees as a part of [online video](https://www.youtube.com/watch?v=LDRbO9a6XPU).
