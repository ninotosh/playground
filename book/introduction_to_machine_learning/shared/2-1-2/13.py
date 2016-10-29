# -*- coding: utf-8 -*-
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report
# RBFカーネルのSVMによる予測モデル構築
clf = svm.SVC()
clf.fit(X_train, y_train)
# クラスラベルの予測
pred = clf.predict(X_test)
# クラスごとのPrecision,Recall,F-値,Accuracyの算出
print(metrics.classification_report(y_test, pred, target_names=['no', 'yes']))
