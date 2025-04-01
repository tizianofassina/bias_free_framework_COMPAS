# Optimized Pre-Processing for Discrimination Prevention




This project aims to offer an alternative for the preprocessing of data to ensure fairness. The experiment is conducted on COMPAS dataset https://projects.propublica.org/datastore/#compas-recidivism-risk-score-data-and-analysis.

We follow the article Optimized Calmon et al, Pre-Processing for Discrimination Prevention, NeurIPS, 2017

## Repository Structure

`cleaning_data.py`
In this file we just clean the data for the experiment. 


`optimization_classical.py`
`optimization_alternative.py`
In this file we implement respective the classical optimization to obtain the transformation. In the alternative we try to implement a new optimization approach based on $\Delta(p_{X,Y}, p_{X, \hat{Y}})$.

`preprocessing_data_classical.py`
`preprocessing_data_alternative.py`
Here we use the obtained classical and alternative distributions to process data
https://github.com/tizianofassina/bias_free_framework_COMPAS.git

## 🔗 Links  
- **Repository GitHub**: [https://github.com/tizianofassina/bias_free_framework_COMPAS.git
](https://github.com/tizianofassina/bias_free_framework_COMPAS.git)
- **Paper originale**: [arXiv:1704.03354 ](https://arxiv.org/abs/1704.03354)
