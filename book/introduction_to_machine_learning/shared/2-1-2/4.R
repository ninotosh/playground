library(kernlab)
# RBFカーネルのサポートベクターマシンによる予測モデルの構築
fit.svm <- ksvm(y~., data=bank.train)
fit.svm
# テストデータに対する予測
pred <- predict(fit.svm, bank.test)
# 予測結果の評価
(conf.mat <- table(pred, bank.test$y)) # 混同行列
(prec <- conf.mat["yes", "yes"]/sum(conf.mat["yes", ])) # 適合率
(rec <- conf.mat["yes", "yes"]/sum(conf.mat[, "yes"])) # 再現率
(f.value <- 2 * prec * rec/(prec + rec)) # F-値
(acc <- sum(diag(conf.mat))/sum(conf.mat)) # 正解率
