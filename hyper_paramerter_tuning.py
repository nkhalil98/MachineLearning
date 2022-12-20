
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

digits = datasets.load_digits()

model_parameters = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'parameters' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'parameters' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'parameters': {
            'C': [1,5,10]
        }
    },
    'naive_bayes_gaussian': {
        'model': GaussianNB(),
        'parameters': {}
    },
    'naive_bayes_multinomial': {
        'model': MultinomialNB(),
        'parameters': {}
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'parameters': {
            'criterion': ['gini','entropy'],
            
        }
    }     
}

scores = []
for model_name, model_parameter in model_parameters.items():
    clf =  GridSearchCV(model_parameter['model'], model_parameter['parameters'], cv=5, return_train_score=False)
    clf.fit(digits.data, digits.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])