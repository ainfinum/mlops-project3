# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

•Random forest classifier trained to predict whether a person makes over 50K a year on Census Income Data Set

•Developed by Andrei Sasinovich in 2021.
 
## Intended Use

Prediction task is to determine whether a person makes over 50K a year.  

## Training Data

The model was trained on publicly available Census Income Data Set.  

The original data set has 32562 rows, and a 80-20 split was used to break this into a train and test set.

To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Evaluation Data

20% of the original data in total of 6512 rows was used for model evaluation. 

One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Metrics

Evaluation metrics include: precision, recall, and F1
 
Model metrics:

Precision: 0.83

Recall: 0.54

F1: 0.65

## Ethical Considerations

Used publicly available Census Bureau data. No new information is inferred or annotated.


## Caveats and Recommendations

An ideal training and evaluation dataset would additionally include years of work experience, position, previous loans, family members details
 