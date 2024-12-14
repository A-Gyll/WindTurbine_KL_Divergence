from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import MultinomialNB

class ClassificationModels:    
    @staticmethod
    def SVMClassifier(C=1.0, kernel='rbf', degree=3, probability=True):
        return SVC(C=C, kernel=kernel, degree=degree, probability=probability)

    @staticmethod
    def RandomForest(max_depth=None, n_estimators=100, n_jobs=-1, min_samples_leaf=1):
        return RandomForestClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            min_samples_leaf=min_samples_leaf
        )

    @staticmethod
    def LRClassifier(C=1.0, penalty='l2', solver='lbfgs', max_iter=100):
        return LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter
        )

    @staticmethod
    def KNeighborsClassifier(n_neighbors=5, leaf_size=30, p=2, n_jobs=-1):
        return KNeighborsClassifier(
            n_neighbors=n_neighbors,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs
        )