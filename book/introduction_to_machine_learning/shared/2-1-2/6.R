# 適合率,再現率,F-値,正解率を評価指標とする評価関数
my.summary <- function(data, lev=NULL, model=NULL) {
  conf <- table(data$pred, data$obs) # 混同行列
  prec <- conf["yes", "yes"]/sum(conf["yes", ]) # 適合率
  rec <- conf["yes", "yes"]/sum(conf[, "yes"]) # 再現率
  f.value <- 2 * prec * rec/(prec + rec) # F-値
  acc <- sum(diag(conf))/sum(conf) # 正解率

  out <- c(Precision=prec, Recall=rec, F=f.value, Accuracy=acc)
  out
}
