library(readr)
library(data.table)
library(xgboost)
library(caretEnsemble)
library(reshape2)
library(dplyr)
library(caret)

########## Starting Code ###########
####################################
total <- read.csv("total.csv")
total <- total[, -1]


### Remove incomplete records ###
total <- total[complete.cases(total), ]
total$ScanCount <- as.numeric(total$ScanCount)
total$VisitNumber <- as.factor(total$VisitNumber)
total$FinelineNumber <- as.factor(total$FinelineNumber)
obs <- nrow(total)


### Choose 80% percent of the data for traning, 20% of the data for testing ###
set.seed(1234)
sample <- sample.int(n = obs, size = floor(0.8*obs), replace = F)
train = total[sample,]
test = total[-sample,]

### Upc Frequncy Table in decreasing order ###
Upcfreq <- as.data.frame(table(total$Upc))
Upcfreq <- Upcfreq[order(-Upcfreq$Freq),]

### To reduce the Upc level, keep the Upc frequency above 25, otherwise, Upc set Upc as "Others" ###
highfreqUpc <- Upcfreq[which(Upcfreq$Freq>25), ][,1]
train$Upc2 <- ifelse(train$Upc %in% highfreqUpc, train$Upc, "Others")
test$Upc2 <- ifelse(test$Upc %in% highfreqUpc, test$Upc, "Others")


data_transform <- function(data){
  data <- data.table(data)

  ### Aggregate the data by Visiting Number ###
  ### Also do some transfermation to make better prediction ###
  x <- data[, list(num_Dept=length(DepartmentDescription),
                   num_FL=length(unique(FinelineNumber)),
                   num_Upc=length(unique(Upc)),
                   TripType = unique(TripType),
                   purchases = sum(ifelse(ScanCount>0,ScanCount,0)),
                   returns = -sum(ifelse(ScanCount<0,ScanCount,0))), by = VisitNumber]

  x <- x[, ':='(fineDeptRatio=num_FL/num_Dept,
                upcDeptRatio=num_Upc/num_Dept,
                upcFineRatio=num_Upc/num_FL,
                returnRatio = returns /(returns+purchases))]
 
  ### Make Change the Levels of DepartmentDescription, Fineline number and Upc number into variables ###
  ### The matrix turned into a sparse matrix ###
  xWeekday <- dcast.data.table(VisitNumber~Weekday, value.var="ScanCount",
                               fun.aggregate = sum, data=data)
  xDept <- dcast.data.table(VisitNumber~DepartmentDescription, value.var="ScanCount",
                            fun.aggregate = sum, data=data)
  xFine <- dcast.data.table(VisitNumber~FinelineNumber, value.var="ScanCount",
                            fun.aggregate = sum, data=data)
  xUpc <- dcast.data.table(VisitNumber~Upc2, value.var="ScanCount",
                           fun.aggregate = sum, data=data)
  
  
  xAgg <- merge(x, xWeekday, by="VisitNumber")
  xAgg <- merge(xAgg, xDept, by="VisitNumber")
  xAgg <- merge(xAgg, xFine, by="VisitNumber")
  xAgg <- merge(xAgg, xUpc, by="VisitNumber")
  
  return(xAgg)
}

### Make transfermations to training data ###
trainTrans <- data_transform(train)
trainTrans <- trainTrans[order(VisitNumber),]

### Make transformations to testing data ###
testTrans <- data_transform(test)
testTrans <- testTrans[order(VisitNumber)]

### Set null value as 0 ###
trainTrans[is.na(trainTrans)] <- 0
testTrans[is.na(testTrans)] <- 0

set.seed(2017)
h <- sample(nrow(trainTrans), 2000)
varnames <- names(which(sapply(trainTrans[, 4:ncol(trainTrans), with=FALSE], function(x) uniqueN(x))>1))

### Remove "TripType" from varnames ###
varnames <- varnames[-2]

train <- trainTrans[, c('TripType', varnames), with=FALSE]

set.seed(1234)
h <- sample(nrow(train), 2000)

### Change the data type into DMatrix that xgboost model can use ###
dval<-xgb.DMatrix(data=data.matrix(train[h,varnames, with=FALSE]),label=data.matrix(train$TripType[h]))
dtrain<-xgb.DMatrix(data=data.matrix(train[-h,varnames, with=FALSE]),label=data.matrix(train$TripType[-h]))

### Use watchlist to measure progress in learning of XGBoost ###
watchlist<-list(val=dval,train=dtrain)

### Set up parameters of XGBoost ###
### eval_metric = "mlogloss" for multi-class classification ###
### num_class = 5 for choosing how many classes the sample have ###
param <- list(objective="multi:softprob",
              eval_metric="mlogloss",
              num_class=5,
              eta = .1,
              max_depth=6,
              min_child_weight=1,
              subsample=1,
              colsample_bytree=1
)

### Fit the XGBoost use xgb.train() ###
### Set maximum number of iterations as 600 ###
### Set print.every.n = 5, print the loss function value every 5 iterations ###
### Set early.stop.round = 50, if the loss function value stays the same for 50 iterations, stop ###
set.seed(1234)
xgb <- xgb.train(data = dtrain,
                  params = param,
                  nrounds = 600,
                  maximize=FALSE,
                  print.every.n = 5,
                  watchlist=watchlist,
                  early.stop.round=50)
save(xgb, file="xgb_split.rda")

### Get the importance of the variables ###
xgbImp <- xgb.importance(feature_names = varnames, model=xgb)

#### Get the information of XGBoost Model ####
xgb

write.csv(xgbImp, "xgbImp_split.csv")

### Create variable names that used in test data set for prediction ###
varnames2 <- names(which(sapply(testTrans[, 4:ncol(testTrans), with=FALSE], function(x) uniqueN(x))>1))
newvarnames <- intersect(varnames, varnames2)

### Transform test data into DMatrix for XGBoost prediction ###
test <- testTrans[, c("TripType",newvarnames), with=FALSE]
Dtest<-xgb.DMatrix(data=data.matrix(test[,newvarnames, with=FALSE]))

### Predict the test data ###
test_pred <- predict(xgb, newdata = Dtest)

### Convert the prediction results of with max probability into labels ###
test_prediction <- matrix(test_pred, nrow = 5, ncol = length(test_pred)/5) %>% 
  t() %>%
  data.frame() %>%
  mutate(label = test$TripType + 1, max_prob = max.col(., "last"))

### Get the Confusion matrix of the real label and prediction results ###
confusionMatrix(factor(test$TripType + 1),
                factor(test_prediction$max_prob),
                mode = "everything")

