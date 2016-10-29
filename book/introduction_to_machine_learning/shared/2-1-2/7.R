library(caret)
set.seed(123)
fit.svm <- train(y~., data=bank.train, method="svmRadial",
  metric="F", tuneGrid=expand.grid(.C=c(0.5, 1.0), .sigma=c(0.05, 0.1)),
  trControl=trainControl(summaryFunction=my.summary, method="cv", number=10))
fit.svm
