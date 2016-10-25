library(data.table)
library(lubridate)
library(caret)
library(xgboost)

setwd("C:/Rajiv/Work/Data Science/Analytics Vidhya/knocktober")

patient <- fread("Patient_Profile.csv")
camp <- fread("Health_Camp_Detail.csv")
first_camp <- fread("First_Health_Camp_Attended.csv")
second_camp <- fread("Second_Health_Camp_Attended.csv")
third_camp <- fread("Third_Health_Camp_Attended.csv")
train <- fread("Train.csv")
test <- fread("Test.csv")

first_camp$Outcome <- 1
first_camp[,":="(Donation=NULL,Health_Score=NULL,V5=NULL)]
second_camp$Outcome <- 1
second_camp[,":="('Health Score'=NULL)]
third_camp$Outcome <- ifelse(third_camp$Number_of_stall_visited>0,1,0)
third_camp[,":="(Number_of_stall_visited=NULL,Last_Stall_Visited_Number=NULL)]
camps <- rbind(first_camp,second_camp,third_camp)
train <- merge(train,patient,by="Patient_ID",all=FALSE)
train <- merge(train,camp,by="Health_Camp_ID",all=FALSE)
train <- merge(train,camps,by=c("Patient_ID","Health_Camp_ID"),all.x=TRUE)
train[is.na(Outcome)]$Outcome <- 0

test <- merge(test,patient,by="Patient_ID",all=FALSE)
test <- merge(test,camp,by="Health_Camp_ID",all=FALSE)
test$Outcome <- -99

full <- rbind(train,test)
full[City_Type==""]$City_Type <- "Other"
full[Employer_Category==""]$Employer_Category <- "Other"

full[,c("Registration_Date",
		"First_Interaction",
		"Camp_Start_Date",
		"Camp_End_Date"
		):=
	list(dmy(Registration_Date),
		dmy(First_Interaction),
		dmy(Camp_Start_Date),
		dmy(Camp_End_Date)
		)]

full$reg_wday <- lubridate::wday(full$Registration_Date)
full[is.na(reg_wday)]$reg_wday <- -99
full$interact_wday <- lubridate::wday(full$First_Interaction)
full$start_wday <- lubridate::wday(full$Camp_Start_Date)
full$end_wday <- lubridate::wday(full$Camp_End_Date)
		
full$Reg_Day <- lubridate::day(full$Registration_Date)
full$Reg_Month <- lubridate::month(full$Registration_Date)		
full$Reg_Year <- lubridate::year(full$Registration_Date)		
full[is.na(Reg_Day)]$Reg_Day <- -99
full[is.na(Reg_Month)]$Reg_Month <- -99
full[is.na(Reg_Year)]$Reg_Year <- -99


full$Interact_Day <- lubridate::day(full$First_Interaction)
full$Interact_Month <- lubridate::month(full$First_Interaction)		
full$Interact_Year <- lubridate::year(full$First_Interaction)		

full$Start_Day <- lubridate::day(full$Camp_Start_Date)
full$Start_Month <- lubridate::month(full$Camp_Start_Date)		
full$Start_Year <- lubridate::year(full$Camp_Start_Date)		

full$End_Day <- lubridate::day(full$Camp_End_Date)
full$End_Month <- lubridate::month(full$Camp_End_Date)		
full$End_Year <- lubridate::year(full$Camp_End_Date)		

full$days_after_interaction <- as.numeric(full$Registration_Date-full$First_Interaction)
full[is.na(days_after_interaction)]$days_after_interaction <- -99
full$days_after_registration <- as.numeric(full$Camp_Start_Date-full$Registration_Date)
full[is.na(days_after_registration)]$days_after_registration <- -99
full$Reg_after_Interaction <- ifelse(is.na(full$Registration_Date),0,1)
full$camp_duration <- as.numeric(full$Camp_End_Date-full$Camp_Start_Date)
		
full[,c("Registration_Date",
		"First_Interaction",
		"Camp_Start_Date",
		"Camp_End_Date",
		"Var1",
		"Var2",
		"Var4",
		"Var5",
		"Income",
		"Education_Score",
		"Age",
		"City_Type",
		"Employer_Category",
		"Category1",
		"Category2",
		"Category3"
		):= 
	list(NULL,
		NULL,
		NULL,
		NULL,
		as.factor(Var1),
		as.factor(Var2),
		as.factor(Var4),
		as.factor(Var5),
		as.factor(Income),
		as.factor(Education_Score),
		as.factor(Age),
		as.factor(City_Type),
		as.factor(Employer_Category),
		as.factor(Category1),
		as.factor(Category2),
		as.factor(Category3)
		)] 

varcol <- sapply(full,is.factor)
fullv <- full[,varcol,with=FALSE]
fulln <- full[,-varcol,with=FALSE]
dummies <- dummyVars(~.,data=fullv)
fullv <- predict(dummies,newdata=fullv)
fullv <- as.data.frame(fullv)
full <- cbind(fulln,fullv)

train <- full[Outcome!=-99]
test <- full[Outcome==-99]
rm(full)
gc()

train_lab <- train$Outcome
train <- train[,":="(Patient_ID=NULL,Health_Camp_ID=NULL,Outcome=NULL)]
test_id1 <- test$Patient_ID
test_id2 <- test$Health_Camp_ID
test <- test[,":="(Patient_ID=NULL,Health_Camp_ID=NULL,Outcome=NULL)]

trainMatrix <- as.matrix(train)
testMatrix <- as.matrix(test)
featureList <- names(train)
#rm(train)
#rm(test)
#gc()

dtrain = xgb.DMatrix(trainMatrix, label=train_lab)
dtest = xgb.DMatrix(testMatrix)

set.seed(1424)
param <- list(objective="binary:logistic",
				eval_metric="auc",
				eta=0.01,
				colsample_bytree = 0.7,
				subsample = 0.7,
				max.depth=7
				)

bst.cv <- xgb.cv(param,dtrain,nrounds=500,nfold=5,early_stopping_rounds=10,print_every_n = 10,verbose= 1)
#nround = which.max(as.matrix(bst.cv)[,3]+as.matrix(bst.cv)[,4])

best <- max(bst.cv$test.auc.mean)
bestIter <- which(bst.cv$test.auc.mean==best)
nround = bestIter*1.5

ensemble <- rep(0, nrow(test))
doTest <- function(param, nround) {
    #watchlist <- list('train' = trainMatrix)
    model = xgboost(
            nrounds = nround
            , params = param
            , data = trainMatrix
			, label=train_lab
            #, watchlist = watchlist
            , print.every.n = 20
            , nthread = 1
            )
    p <- predict(model, testMatrix)
    rm(model)
    gc()
    p
}

for (i in 1:3) {
    print(i)
    set.seed(i + 2017)
    p <- doTest(param, nround) 
    ensemble <- ensemble + p
}
xgb_pred_test <- ensemble/i

#xgb_mod <- xgboost(param=param, data=dtrain,nrounds=1000, verbose=1,print.every.n = 5)


# featureVector <- c() 
# for (i in 1:length(featureList)) { 
  # featureVector[i] <- paste(i-1, featureList[i], "q", sep="\t") 
# }

# write.table(featureVector, "fmap.txt", row.names=FALSE, quote = FALSE, col.names = FALSE)
# xgb.dump(model = xgb_mod, fname = 'xgb_knock.dump', fmap = "fmap.txt", with.stats = TRUE)


#xgb_pred <- predict(xgb_mod,newdata=testMatrix)

#my_solution <- data.frame(Patient_ID=test_id1,Health_Camp_ID=test_id2,Outcome=xgb_pred)
my_solution <- data.frame(Patient_ID=test_id1,Health_Camp_ID=test_id2,Outcome=xgb_pred_test)

# Write your solution away to a csv file with the name my_solution.csv
#write.csv(my_solution,file="xgb.csv",row.names=FALSE)
#write.csv(my_solution,file="xgb_one_hot_encoding.csv",row.names=FALSE)
#write.csv(my_solution,file="xgb_new_features.csv",row.names=FALSE)
#write.csv(my_solution,file="xgb_more_nrounds.csv",row.names=FALSE)
write.csv(my_solution,file="xgb_ensemble.csv",row.names=FALSE)