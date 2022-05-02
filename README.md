# Scored [7th Place](https://www.drivendata.org/competitions/83/cloud-cover/leaderboard/) in the On Cloud N: Cloud Cover Detection Challenge

My main ideas that got me a high score were
- Significant cleaning of the dataset based on model disagreement and manual visual inspection
- Ensembling many models trained on different subsets of the data and different hyperparameter choices
- Exporting the trained models to JIT, making them more efficient and allowing me to use more models for the final predictions in the drivendata testing environment