# caret パッケージで使用可能なアルゴリズム
head(modelLookup())
# 使用可能なアルゴリズム数
packageVersion("caret")
length(unique(modelLookup()$model))
