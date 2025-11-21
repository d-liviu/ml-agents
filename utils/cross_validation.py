import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

# depends on what file we are reading

df = pd.read_csv("example_output.csv",).drop(columns=["run_id","algo_name"])
df_worm = df[df["env_name"] == "Worm"].copy()
df_pyramids = df[df["env_name"] == "Pyramids"].copy()

# do NOT use all metrics --> e.g. runID is useless, turn categorical metrics (e.g. algo_name) to numerical values
# and then use the metrics

# X = metrics and values
# Y = classification -> episodic_reward_mean
X_worm = df_worm.iloc[:, :-1].values 
y_worm = df_worm.iloc[:, -1].values 

X_pyramids = df_pyramids.iloc[:, :-1].values 
y_pyramids = df_pyramids.iloc[:, -1].values 

#helper method: yelding test train splits as arrays of indeces as found in teh guide:
def custom_4folds(X, k):
    data_amount= X.shape[0]  #checked with X.shape and returns correct tuple : (477, 211)
    Data_index = np.arange(data_amount)
    #shuffling the data to avoid inbalance
    np.random.shuffle(Data_index)
    fold = data_amount // k
    for i in range(k):
        test_idx = Data_index[i*fold:(i+1)*fold]
        train_idx = np.concatenate((Data_index[:i*fold], Data_index[(i+1)*fold:]))
        yield train_idx, test_idx

#getting the cross validation score. decided to go for the each test split = 0.25 /0.75--> 4 folds
def crossVal_score(clf, X, Y, k):
    result_scores = []
    for train_idx, test_idx in custom_4folds(X, k):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        
        forest = forest_maker(X_train, y_train, 50)
        y_pred = forest_prediction(forest, X_test)
        accuracy_perfold= np.mean(y_pred==y_test)
        result_scores.append(accuracy_perfold)

    return np.array(result_scores)

#forest creation: I chose 50 trees (se crossval_scre function) because it gets enough diversity to see the effect and it's also stable
def forest_maker(X,y,n_trees):
    n = X.shape[0]
    forest = []
    for i in range(n_trees):
        sample_idx = np.random.choice(n, n, replace=True)
        X_sample, y_sample = X[sample_idx], y[sample_idx]
        tree = DecisionTreeClassifier(random_state=i)
        tree.fit(X_sample, y_sample)
        forest.append(tree)

    return forest

#forest prediction: when predicting, each tree votes, and the majority wins
def forest_prediction(forest, X_test):
    predictions = []
    for tree in forest:
        predictions.append(tree.predict(X_test))
    predictions = np.array(predictions)
    result = stats.mode(predictions, axis=0, keepdims=False).mode
    return result


#using all helper methods above to actually get predictions:
clf = DecisionTreeClassifier(random_state=0)
k=4  
scores_worm = crossVal_score(clf, X_worm, y_worm, k)
scores_pyramids = crossVal_score(clf,X_pyramids,y_pyramids,k)
print(scores_worm)
print(scores_pyramids)
print("Average accuracy for Worm enviroment: ", scores_worm.mean())
print("Average accuracy for Pyramids enviroment: ", scores_pyramids.mean())
