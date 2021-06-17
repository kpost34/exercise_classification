#1-Load Packages
library("caret")
library("tidyverse")
library("lubridate")
library("randomForest")

#2-Get Data 
setwd("/Users/keithpost/Documents/Coursera/Data Science Specialization/8-Practical Machine Learning/exercise_classification/data/")
fileUrl<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(fileUrl,destfile="traindata.csv",method="curl")
fileUrl2<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(fileUrl2,destfile="testdata.csv",method="curl")


#3-Create Training and Testing Data Frames
training<-read.csv("traindata.csv")
testing<-read.csv("testdata.csv")


#4-Clean Data
#See dimensions of data frames
dim(training) #19622 x 160
dim(testing) #20 x 160

#Trim training data frame
summary(training) #many variables contain lots of NAs (i.e., 19216/19622)
sum(colSums(is.na(training))==19216) #67 variable contain ~98% NAs
training2<-training[,which(colSums(is.na(training))!=19216)] #removes variables with excess missingness and stores into new data frame
head(training2) #notice missing data (not marked as NAs)
training2<-training2 %>% mutate_all(na_if,"") #converts missing data to NAs
summary(training2) #identify more variables with excess NAs
fulltrain<-training2[,which(colSums(is.na(training2))!=19216)] #removes variables with excess NAs
sum(is.na(fulltrain)) #0; no remaining variables contain NAs

#Convert date-time variable to appropriate format
fulltrain$cvtd_timestamp<-strptime(fulltrain$cvtd_timestamp,format="%d/%m/%Y %H:%M")
fulltrain$cvtd_timestamp<-as.numeric(fulltrain$cvtd_timestamp)

#Drop first column (row label)
fulltrain<-fulltrain[,-1]

#Create probe data
set.seed(34)
inTrain<-createDataPartition(y=fulltrain$classe,p=0.7,list=FALSE)
trainEx<-fulltrain[inTrain,]
probeEx<-fulltrain[-inTrain,]
dim(trainEx) #13737 x 59
dim(probeEx) #5885 x 59


#5-Exploratory Data Analysis
#Basic bivariate plots
qplot(gyros_arm_x,gyros_arm_y,data=trainEx)
qplot(gyros_belt_z,gyros_belt_y,data=trainEx)
#clear multicollinearity

#Some bivariate scatter plots with classe added as color
qplot(gyros_belt_x,accel_belt_x,data=trainEx,color=classe)
qplot(gyros_arm_z,accel_arm_z,data=trainEx,color=classe)
#no clear patterns by classe; patterns seem to be obscured by purple (classe E, which is plotted last)

#Bivariate plots by user
qplot(gyros_belt_z,accel_belt_z,data=trainEx,color=classe,facets=.~user_name,alpha=0.1)
qplot(gyros_belt_x,accel_belt_x,data=trainEx,color=classe,facets=.~user_name,alpha=0.1)
#same issue--data obscured by classe E

#Faceted plots by classe (to see any patterns by classe)
qplot(gyros_belt_z,accel_belt_z,data=trainEx,color=classe,facets=.~classe)
qplot(roll_belt,pitch_belt,data=trainEx,color=classe,facets=.~classe)
#there appear to be bivariate patterns for each classe; however, could these be separated into clear groups using a classifier?

#Data standardization
mean(trainEx[,8]); sd(trainEx[,8])
mean(trainEx[,9]); sd(trainEx[,9])
#sd is substantially larger than means in these examples

#Standardize data (not columns 1, 5, or 59)
preObj<-preProcess(trainEx[,-c(1,5,59)],method=c("center","scale"))
trainPreProc<-predict(preObj,trainEx[,-c(1,5,59)])
summary(trainPreProc)
trainPrepEx<-cbind(trainEx[,c(1,5)],trainPreProc,classe=trainEx[,59])

#Bivariate plots of normalized data
qplot(gyros_belt_z,accel_belt_z,data=trainPrepEx,color=classe,facets=.~classe)
qplot(roll_belt,pitch_belt,data=trainPrepEx,color=classe,facets=.~classe)
#don't notice much (if any) difference with or without standardizing data


#6-PCA to Reduce Dimensionality of Data
#Training data
#using prcomp to understand proportion of variance
prComp<-prcomp(trainEx[,-c(1,5,59)],center=TRUE,scale=TRUE)
summary(prComp) #eigenvalue hits 1 at ~PC14 (~81% variance explained)
screeplot(prComp,npcs=40,type="l") 
abline(h=1,col="red") #scree plot; shows where ev=1

#produce training data PCs using preProcess at a cut-off of 0.81 (81% variance)
preProc<-preProcess(trainEx[,-59],method="pca",thresh=.81)
trainPC<-predict(preProc,trainEx[,-59])
trainPC<-cbind(trainPC,classe=trainEx[,59])

#apply the same process to probe data
probePC<-predict(preProc,probeEx[,-59])


#7-Fit Models with Training Data
#Recursive partitioning
modelTree<-train(classe~.,method="rpart",data=trainPC)
confusionMatrix(trainEx$classe,predict(modelTree))$overall[1] #0.4174

#Random forest
modelRF<-randomForest(classe~.,data=trainPC)
confusionMatrix(trainEx$classe,predict(modelRF))$overall[1] #0.9656

#Linear discriminant analysis
modelLDA<-train(classe~.,method="lda",data=trainPC)
confusionMatrix(trainEx$classe,predict(modelLDA))$overall[1] #0.5156

#Bagging
modelBag<-train(classe~.,data=trainPC,method="treebag")
confusionMatrix(trainEx$classe,predict(modelBag))$overall[1] #0.9998


#8-Assess Out of Sample Error with Selected Model
#Random forest
confusionMatrix(probeEx$classe,predict(modelRF,probePC)) 

#Bagging
confusionMatrix(probeEx$classe,predict(modelBag,probePC)) 


#9-Clean Test Data Set
#Trim testing data frame
summary(testing) #many variables with 20/20 NAs
testing2<-testing[,which(colSums(is.na(testing))!=20)] #remove variables without data and store in new object
testEx<-testing2[,-1] #drop first variable (row numbers)
sum(complete.cases(testEx)) #all rows have complete cases
colnames(testEx) %in% colnames(trainEx) #all columns match (except the last one, problem id)

#Convert date-time variable to appropriate format
testEx$cvtd_timestamp<-strptime(testEx$cvtd_timestamp,format="%d/%m/%Y %H:%M")
testEx$cvtd_timestamp<-as.numeric(testEx$cvtd_timestamp)


#10-PCA on Testing Data
#Run training PCs on testing data
testPC<-predict(preProc,testEx[,-59])

#Trick to circumvent error when using RF model to predict classe values of testing data
trainPCnew<-trainPC[,-17]
testPC<-rbind(trainPCnew[1,],testPC)
testPC<-testPC[-1,]
rownames(testPC)<-1:20


#11-Predict classe Values of Testing Data
predict(modelRF,testPC)






