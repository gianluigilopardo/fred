## Faithful and Robust Local Interpretability for Textual Predictions
This repository contains the official implementation of 
> Lopardo, G., Precioso, F., & Garreau, D. "[Faithful and Robust Local Interpretability for Textual Predictions](https://arxiv.org/abs/2311.01605)." 

The original FRED code and the experiments presented in the paper are both available here. 

### Installation

```
pip install requirements.txt
python -m spacy download en_core_web_lg
```


## Usage
### Experiments

To replicate the experiments, simply run: 

```python3 main.py --dataset DATASET --model MODEL```

* DATASET: restaraunts, yelp 
* MODEL: logistic_classifier, forest_classifier, roberta

The code will then compare the FRED, LIME, and Anchors explainers on the given dataset and model, evaluating them on faithfulness, robustness, time, and the proportion of the document used for explainability. 

Results will appear in the directory ```results```. 



### FRED
If you just want to apply FRED to explain your model ```model``` on a document ```doc```, run 

```
from fred import explainer

explainer = explainer.Fred(class_names=class_names, classifier_fn=model.predict_proba)
exp = explainer.explain_instance(doc)
print(exp.best)
```


### Bibtex
```
@article{lopardo2023faithful,
  title={Faithful and Robust Local Interpretability for Textual Predictions},
  author={Lopardo, Gianluigi and Precioso, Frederic and Garreau, Damien},
  journal={arXiv preprint arXiv:2311.01605},
  year={2023}
}
```
