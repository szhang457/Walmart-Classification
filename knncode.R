library(gmodels)
library(data.table)
#read date
t1 <- data.table(read.csv("Data_Trans_2.csv"))
#select 5 trip types
myData <- t1[class %in% c('29','30','17','4','5'), ]
myData<-as.matrix(myData)
Data<-myData[,-c(1,78)]
#randomly choose train data and test data
m<-nrow(Data)
val<-sample(1:m, size = round(m/4), replace = FALSE, prob = rep(1/m, m)) 
Train<-Data[-val,]
Test<-Data[val,]
#choose an appropriate value for K
kk<-seq(1,30,by=1)
accu<-matrix(0,1,30)
for(i in 1:30){
pre_result<-knn(train=Train[,2:76],test=Test[,2:76],cl=Train[,1],k=kk[i])
summary(pre_result)
#compute accuracy
ac<-table(pre_result,Test[,1])
acc<-sum(diag(ac))
accu[1,i]<-acc/sum(ac)
}
show(accu)
#plot relationship between accuracy score and k
plot(x=kk,y=accu,xlab="k",ylab="Accuracy")

#plot heatmap
library(pheatmap)
pheatmap(ac,clustering_distance_rows = "correlation",clustering_method = "complete",color=colorRampPalette(c("green","red"))(100),scale = "row",margins=c(30,30),fontsize_row = 8,cellheight = 30,cellwidth = 30)
write.table(ac,file="table.csv")

