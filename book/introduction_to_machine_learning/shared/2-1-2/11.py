# -*- coding: utf-8 -*-
import pandas as pd
# 特徴量を構築する関数
def makefeature(x):
    # 数値変数のスケーリング
    cn_num = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    x_num = x[cn_num]
    x[cn_num] = (x_num - x_num.mean())/x_num.std()
    # ダミー変数への変換
    x_dum = pd.get_dummies(x)
    return x_dum
