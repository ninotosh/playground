# 前処理を実行する関数
makefeature <- function(x)
{
  is.num <- sapply(x, class) == "integer"
  x[, is.num] <- lapply(x[, is.num], scale)

  x
}
