library(readr)
library(data.table)
library(xgboost)
library(caretEnsemble)
library(reshape2)
library(dplyr)
library(proxy)
library(qlcMatrix)
library(cccd)
library(igraph)
t1 <- data.table(read.csv("train.csv"))
#weekday coding
WeekdayClasses <- data.frame(Weekday=c("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"), WeekdayNumber=seq(1,7))
t1 <- merge(t1, WeekdayClasses, by="Weekday")
t1 <- t1[order(t1$VisitNumber),]
t1$Weekday <- NULL
#triptype coding
tripClasses <- data.frame(TripType=sort(unique(t1$TripType)), class=seq(0,37))
t1 <- merge(t1, tripClasses, by="TripType")
t1 <- t1[order(t1$VisitNumber),]
TripType <- t1$TripType
t1$TripType <- NULL
#return variable
t1$Returns <- -t1$ScanCount
t1$Returns[t1$Returns < 0] <- 0
t1$Purchases <- t1$ScanCount
t1$Purchases[t1$Purchases < 0] <- 0
#topUPC
topUPC <- names(sort(which(table(t1$Upc)>200), decreasing=TRUE))
t1$TopUpc <- ifelse(t1$Upc %in% topUPC, 1, 0)

#merge data
#SQL complete
write_csv(t1, "Data_Trans.csv",";")

t2 <- data.table(read.csv("Data_Trans_1.csv"))
#department coding
xDept <- dcast.data.table(VisitNumber~DepartmentDescription, value.var="ScanCount", fun.aggregate = sum, data=t1)
t2 <- merge(t2, xDept, by="VisitNumber")
t2 <- t2[order(t2$VisitNumber),]
t2[is.na(t2)]<-0
#department count
myFun=function(x,c1,c2){
count=0
for(i in c1:c2){ 
if(x[i]!=0){
count=count+1
}
}
return(count)
}
t2$DepartCount=apply(t2,1,myFun,c1=7,c2=75)
#ratio
t2$upcdepart=t2$UpcCount/t2$DepartCount
write_csv(t2, "Data_Trans_2.csv",";")
