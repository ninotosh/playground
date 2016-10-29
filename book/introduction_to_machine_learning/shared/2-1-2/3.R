set.seed(123)
# データの読み込み
bank <- read.csv("data/bank-full.csv", sep=";")
head(bank, 3)
# 各列のデータ型の確認
sapply(bank, class)
# 前処理の実行
bank.processed <- makefeature(bank)
# 訓練データとテストデータの作成
N <- nrow(bank)
inds.tr <- sample(seq(N), as.integer(0.7 * N))
bank.train <- bank.processed[inds.tr, ]
bank.test <- bank.processed[-inds.tr, ]
