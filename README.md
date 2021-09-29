# Fundus-Descriptor

## Requirements
- Python >=3.7

```bash
conda create -n fundus_descriptor python=3.7
conda activate fundus_descriptor
```

## Setup environment
Run this script to create a virtual environment and install dependency libraries

```bash
pip install -r requirements.txt
```

#Download data
```bash
bash setup_data.sh
```

#Train model
Fill in the config/config.json file the required configuration and run:
```python
python train.py
```
Expected Output:
```
Epoch: 0
        Training batch 1 Loss: 19.729628
        Training batch 2 Loss: 15.866058
        Training batch 3 Loss: 17.826273
        Training batch 4 Loss: 17.788681
        Training batch 5 Loss: 42.941818
        Training batch 6 Loss: 22.389496
        Training batch 7 Loss: 47.479721
        Training batch 8 Loss: 36.725548
        Training batch 9 Loss: 45.582085
        Training batch 10 Loss: 35.621620
        Training batch 11 Loss: 34.154144
        Training batch 12 Loss: 40.838852
        Training batch 13 Loss: 12.401539
        Training batch 14 Loss: 45.432705
        Training batch 15 Loss: 40.167793
        Training batch 16 Loss: 32.452755
        Training batch 17 Loss: 56.421425
        Training batch 18 Loss: 46.686169
        Training batch 19 Loss: 43.404442
        Training batch 20 Loss: 33.464497
        Training batch 21 Loss: 49.079994
        Training batch 22 Loss: 64.510696
        Training batch 23 Loss: 40.696648
Training set: Average loss: 36.594025, Average accuracy: 93.879%, AUC score: 0.993

Validation set: Average loss: 182.907195, Average accuracy: 87.026%, AUC score: 0.972

Validation accuracy increased (inf --> 0.870).  Saving model to ./models/trial-80.pth
```

#Test model 
```python
python test.py
```
Expected Output:
```
Validation set: Average loss: 84.072955, Average accuracy: 90.961%, AUC score: 0.983

accuracy_list    [0.95758929 0.90104167 0.87331536 0.84175084 0.80272109 0.89595376]
precision_list   [0.9862069  0.93010753 0.94186047 0.90252708 0.93650794 0.95975232]
recall_list      [0.97058824 0.96648045 0.92307692 0.92592593 0.84892086 0.93093093]
f1_list  [0.97833523 0.94794521 0.9323741  0.91407678 0.89056604 0.94512195]
sensitivity_list         [0.97058824 0.96648045 0.92307692 0.92592593 0.84892086 0.93093093]
specificity_list         [0.96648045 0.97058824 0.92592593 0.92307692 0.98340249 0.95486111]
auc_score_list   [0.99560151 0.99555095 0.9699905  0.96986916 0.98189498 0.98658033]
accuracy mean    0.9096081588835212
precision mean   0.9428270371418931
recall mean      0.9276538875774373
f1_score micro   0.9421439060205581
f1_score macro   0.9347365518925649
sensitivity mean         0.9276538875774373
specitivity mean         0.9540558553270012
auc_score mean   0.9832479047209327
```

## Predict image
Fill in the config/config_predict.json file the required configuration and run:
```python
python predict.py
```

The `result` is a dictionary with the structure:
```javascript
{'prob_central': 0.9996613264083862, // Probability of fundus image is of central area 
 'prob_peripheral': 0.0002774616295937449, // Probability of image is of peripheral area
 'prob_left': 9.435890387976542e-05,  // Probability of image is of left eye
 'prob_right': 0.9999237060546875, // Probability of image is of right eye
 'prob_od': 0.0004447030369192362,  // Probability of image is optic-disc-centered
 'prob_macula': 0.9996280670166016,  // Probability of image is macula-centered
 'label': ['central', 'right', 'macula'] // Label from classifier: 'left' or 'right' or 'undetermined'
}
```
