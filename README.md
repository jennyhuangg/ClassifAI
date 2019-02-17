# TreeHacks 2019 Project
Applies Classifier Learning Techniques On Twitter Data To Predict Mental Health Risk

## To Run Algorithms:

```
python test_gnb.py # gaussian naive bayes
python test_mnb.py # multinomial naive bayes
python test_knn.py # k-nearest neighbors
```

## To Generate Data (which has already been done):

### Installations

#### Twitter API
Download the Python twitter tools at https://pypi.python.org/pypi/twitter. Then, run
```
python setup.py build     
python setup.py install
```

#### Sentiment Analysis Libraries
```
pip install textblob
python -m textblob.download_corpora
```
### Re-generate Data
Note: This has already been done and the data files have been generated in the repo. 
```
python generatedata/search_names.py
python generatedata/search_timelines.py
python generatedata/extract_features.py # gaussian NB
python generatedata/extract_text.py # binomial NB and K-NN
```


