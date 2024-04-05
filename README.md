## Faithful and Robust Local Interpretability for Textual Predictions
This repository contains the official implementation of 
> Lopardo, G., Precioso, F., & Garreau, D. "[Faithful and Robust Local Interpretability for Textual Predictions](https://arxiv.org/abs/2311.01605)." 


### Installation

```
pip install requirements.txt
python -m spacy download en_core_web_lg
```


## Usage
### Experiments

To replicate the experiments, simply run: 

```python3 main.py --dataset DATASET --model MODEL```

* DATASET: restaraunts, yelp, tweets, imdb
* MODEL: logistic_classifier, tree_classifier, forest_classifier, distilbert, roberta

The code will then compare the FRED, LIME, SHAP, and Anchors explainers on the given dataset and model, evaluating them on faithfulness, robustness, time, and the proportion of the document used for explainability. 

Results will appear in the directory ```results```. 



### FRED
If you just want to apply FRED to explain your model ```model``` on a document ```doc```, run 

```
from fred import explainer

explainer = explainer.Fred(class_names=class_names, classifier_fn=model.predict_proba)
exp = explainer.explain_instance(doc)
print(exp.best)
```

See ```fred_example.ipynb``` and ```fred_saliency.ipynb``` for counterfactuals and saliency weights tutorials. 

