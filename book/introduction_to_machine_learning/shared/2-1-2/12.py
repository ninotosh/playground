# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cross_validation import train_test_split
# データの読み込み
bank = pd.read_csv("data/bank-full.csv", sep=";")
print(bank.head(3))
features, label = makefeature(bank.drop('y', 1)), bank.y
# 訓練データとテストデータの作成
random_state = np.random.RandomState(123)
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=.3, random_state=random_state)
