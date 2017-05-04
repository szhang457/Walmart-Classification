t1 <- data.table(read.csv("Data_Trans_2.csv"))
s1 <- data.table(read.csv("test.csv"))
freq[order(-freq[,1]),]

total <- t1[TripClass %in% c('29','30','17','4','5'), ]
write_csv(total, "Data_Trans_3.csv")