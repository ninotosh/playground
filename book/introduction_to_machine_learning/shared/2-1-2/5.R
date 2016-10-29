library(randomForest)
set.seed(123)
# ランダムフォレストによる予測モデルの構築
fit.rf <- randomForest(y~., data=bank.train)
fit.rf
# テストデータに対する予測
pred <- predict(fit.rf, bank.test)
# 予測結果の評価
(conf.mat <- table(pred, bank.test$y)) # 混同行列
(prec <- conf.mat["yes", "yes"]/sum(conf.mat["yes", ])) # 適合率
(rec <- conf.mat["yes", "yes"]/sum(conf.mat[, "yes"])) # 再現率
(f.value <- 2 * prec * rec/(prec + rec)) # F-値
(acc <- sum(diag(conf.mat))/sum(conf.mat)) # 正解率
