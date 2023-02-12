# Setup
1. Clone the repo
```
git clone https://github.com/steviedale/chess_vision.git
```

2. Create a virtual environment
```
python3 -m venv venv
```

3. Source virtual environment
```
source venv/bin/activate
```

4. Install pip packages
```
pip install -r requirements.txt
```

5. Rename absolute paths in .csv file.
    1. Go to dataset/dataframes/train.csv
    2. find/replace "/home/stevie/git/chess_vision" with the path to this repo
    3. repeat these steps for dataset/dataframes/test.csv

# Run Experiments

## KNN
1. run the jupyter notebook at knn/knn.ipynb 
2. view results in the knn/results folder (there should be four graphs: training time, eval time, train accuracy, and test accuracy)

## Decision Tree
1. run the jupyter notebook at decision_tree/decision_tree.ipynb 
2. view results in the decision_tree/results folder (there should be four graphs: training time, eval time, train accuracy, and test accuracy)

## Boosting
1. run the jupyter notebook at boosting/boosting.ipynb 
2. view results in the boosting/results folder (there should be four graphs: training time, eval time, train accuracy, and test accuracy)

## SVM
1. run the jupyter notebook at svm/svm.ipynb 
2. view results in the svm/results folder (there should be four graphs: training time, eval time, train accuracy, and test accuracy)

## Neural Networks
1. source the virtual environment
```
source venv/bin/activate
```
2. change directory to neural_network dir
```
cd neural_network
```
3. run training
```
python train.py
```
Note: I used weights and biases to log my learning curves. Unfortunately I can't give you access to my wandb. But you are welcome to
update the commented out code that refers to wandb and add in your own wandb info or another logger.