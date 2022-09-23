source venv/bin/activate
cd boosting
python boosting_cross_val.py
cd ../decision_tree
python decision_tree_cross_val.py
cd ../knn
python knn.py
python knn_cross_val.py
