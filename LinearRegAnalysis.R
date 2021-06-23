#Import Data
data <- read.csv(file.choose(), header=T)
#derive separate data 
set.seed(100478003)
rows <- sample(1:nrow(data), 1000, replace = FALSE)
housing <- data[rows,]

str(housing)
set.seed(1)

#Separate data into train and test groups
#build model on train dataset 
train <- housing[sample(1:nrow(housing), 500, replace=F), ] #sample from observations (1: rows), split in ~half (100 for model, 96 for validate)
test <- housing[which(!(housing$X %in% train$X)),]
write.csv(train, file="housing_train.csv")
write.csv(test, file="housing_test.csv")

#Look at full model
pairs(train[,-1])  #shows that we have highly skewed variables, large number of small values and very few large values at many predictors 
full <- lm(median_house_value ~ ., data=train[,-1]) #remove the first column, the identifier
summary(full)
install.packages("gamlr")
library(gamlr)
AIC(full)
BIC(full)
AICc(full)
#a lot of significance, only NOT sig at total rooms 
#pretty good R^2, 0.5863 and Fstat
#strong p value 

#Residuals and plots to validate assumptions
par(mfrow=c(3,4))
plot(rstandard(full)~fitted(full), xlab="fitted", ylab="Residuals", main = "Residual Plot")
for(i in c(2:10)){ #without 1, 
  plot(rstandard(full)~train[,i], xlab=names(train)[i], ylab="Residuals", main = "Residual Plot")
}
qqnorm(rstandard(full))
qqline(rstandard(full))
plot((train$median_house_value)~fitted(full), ylab = "Median House Value", main = "Response vs. Fitted Values")
abline(a=0,b=1)
lines(lowess((train$median_house_value)~fitted(full)), col="blue", xlab = "median house value")
#Predictor resdiauls are skewed to the small x values 

#####################################
####### CONDITIONS ##############
####################################
#Check conditions to see if residual plots can tell us what is wrong with the model 

#Condition 2: Make scatterplot of predictors to see pairwise relationships 

pairs(train[,2:10], lower.panel = NULL, main = "Scatterplot of Continuous Variables") #for just continuous variables 
#we are looking for patterns: 
#longitude and latitude looks weird 
#concentrated at smaller X values
#BUT no curves or bends, so we will say it looks fine


#Check condition 1:Plot of response against fitted values 
plot(train$median_house_value~fitted(full), main="Y v Fitted", xlab="Fitted", ylab="median house value")
abline(a = 0, b = 1)
lines(lowess(train$median_house_value~fitted(full)), col="blue")
#Condition 1 fails: not identity function because not straight 
#not horrible but it looks like theres two curves (sin curve)
#weird points at high median house value could be affecting regression
#points heavily concentrated at low med. house values 

#Since condition 1 fails, we cannot use our residual plots to discover what is wrong with out model
#or how to fix it

install.packages("alr4")  
library(alr4)

#####################################
####### TRANSFORMATION ##############
####################################
apply(housing[,2:11], 2, hist, ) #heavy skew
hist(train$median_house_value, xlab = "median housing price", main = "Distribution of Response Before Log Transformation")
#Don't transform indicator variables, only continuous variables
#BoxCox for variables
mult <- lm(cbind(housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value) ~ 1, data = train)
bc <- powerTransform(mult)
summary(bc)

#transform only the response 
par(mfrow=c(2,1))
hist(train$median_house_value, xlab = "Median Housing Value", main = "Distribution of Response Before Log Transformation")
train$median_house_value <- log(train$median_house_value) #because 0 in CI, and rounded power = 0 
hist(train$median_house_value, xlab = "Median Housing Value", main = "Distribution of Response After Log Transformation")

#Create new model with transformed response and excluding inland 
new_mod <- lm(median_house_value ~ ., data=train[,-c(1,14)])
summary(new_mod)
AIC(new_mod)
BIC(new_mod)
AICc(new_mod)
#Check assumptions: residual plots and QQnorm 
par(mfrow=c(4,4))
plot(rstandard(new_mod)~fitted(new_mod), xlab="fitted", ylab="Residuals")
for(i in c(2:14)){ #without 1, 
  plot(rstandard(new_mod)~train[,i], xlab=names(train)[i], ylab="Residuals")
}
qqnorm(rstandard(new_mod))
qqline(rstandard(new_mod))
plot((train$median_house_value)~fitted(new_mod))
abline(a=0,b=1)
lines(lowess((train$median_house_value)~fitted(new_mod)), col="blue", xlab = "median housing value")

#Compare transformed model to full model to check improvements
AIC(full)
AIC(new_mod) #significantly lowers AIC and BIC and raises r^2 
BIC(full)
BIC(new_mod)
summary(full)$adj.r.squared
summary(new_mod)$adj.r.squared

#####################################
####### MULTICOLINEARITY ############
####################################

#Check correlation of variables excluding case 
cor(train[,-1], lower.panel = FALSE)
# rather than do it individually, let's make a table
install.packages("xtable")  # only if not installed
library(xtable)   # you may need to go to Tools > Install Packages and load xtable
mcor <- round(cor(train[,-1]), 2) #take cor of all variables in data
mcor #correlation matrix
upper <- mcor #isolate top corner
upper[upper.tri(mcor)] <- ""   # cells in upper triangle of matrix are deleted
upper <- as.data.frame(upper) # save it as a data frame (easier to work with)
upper
write.csv(upper, "test.csv")
#long and latitude correlated
#total bedrooms and total rooms and population and households
#INLAND IS NA SO ITS PERFECTLY CORRELATED -> DELETE

#Check VIFs of new_mod
vif(new_mod)
#VIFs greater than 5: longitude, latitude, total rooms, bedrooms, population, households

#look closely at highest VIFs: Total Bedrooms 
TotalBRmod <- lm(total_bedrooms ~ ., data=train[,-c(1,10,14)]) #dont include name identifier or response
summary(TotalBRmod)
#very high R^2 
#total bedrooms is colinear with it 

#look closely at highest VIFs: Latitude 
latmod <- lm(latitude ~ ., data=train[,-c(1,10,14)]) #dont include name identifier or response
summary(latmod) 

#Transformed model with total bedrooms removed 
full1 <- lm(median_house_value ~ ., data=train[,-c(1,14,6)]) #without totalbedrooms (6)
summary(full1)
vif(full1) #better but still high for rooms and bedrooms so remove total rooms 

#Transformed model Without total rooms 
full2 <- lm(median_house_value ~ ., data=train[,-c(1,14,6,5)]) 
summary(full2)
vif(full2) 

#better but still issues with long and lat.
#Transformed model Without Latitude 
full3 <- lm(median_house_value ~ ., data=train[,-c(1,14,6,5,3)]) 
summary(full3)
vif(full3) #then longitude loses significance! (Because correlated with latitude )
AIC(full3)
AICc(full3)
BIC(full3)
# check assumptions
par(mfrow=c(3,4))
plot(rstandard(full3)~fitted(full3), xlab="fitted", ylab="Residuals")
for(i in c(2, 4:5, 7,9:13)){ #without 1, 3, 6, 8, 14
  plot(rstandard(full3)~train[,i], xlab=names(train)[i], ylab="Residuals")
}
qqnorm(rstandard(full3))
qqline(rstandard(full3))
plot((train$median_house_value)~fitted(full3))
abline(a=0,b=1)
lines(lowess((train$median_house_value)~fitted(full3)), col="blue")
# everything still looks ok 

##########################################################################
    ####### LEVERAGE, OUTLIERS, AND INFLUENTIAL POINTS ############
#########################################################################
library(plyr)

#leverage
h <- hatvalues(full3)
hcut <- 2*length(coefficients(full3))/nrow(train)
w1 <- which(h > hcut)
w1

#outliers
r <- rstandard(full3)
w2 <- which(r >=2 | r <= -2)
w2
#so many

# check for influence
d <- cooks.distance(full3)
dcut <- qf(0.5, length(coef(full3)), full3$df.residual) #use f distribution
w3 <- which(d > dcut)
w3
#none

#influence your own fitted value 
dffit <- dffits(full3)
cutfit <- 2*sqrt(length(coef(full3))/nrow(train))
w4 <- which(abs(dffit) > cutfit)
w4
#so many 

dfbeta <- dfbetas(full3)
cutb <- 2/sqrt(nrow(train))
w5 <- which(abs(dfbeta[,1]) > cutb | abs(dfbeta[,2]) > cutb | abs(dfbeta[,3]) > cutb | abs(dfbeta[,4]) > cutb| abs(dfbeta[,5]) > cutb| abs(dfbeta[,6]) > cutb| abs(dfbeta[,7]) > cutb| abs(dfbeta[,8]) > cutb| abs(dfbeta[,9]) > cutb)
w5
#so many 
#since so many data points, wont really influence model 


par(mfrow=c(3,3))
plot((train[,10])~train[,9], main="Median Housing v Median Income", xlab="Median Income", ylab="log(median house value)")
points((train[w1,10])~train[w1,9], col="red", pch=16)
points((train[w2,10])~train[w2,9], col="red", pch=16)
points((train[c(w4,w5),10])~train[c(w4,w5),9], col="red", pch=16)
plot((train[,10])~train[,11], main="Median Housing v Near Bay", xlab="Near Bay", ylab="log(median house value)")
points((train[w1,10])~train[w1,11], col="red", pch=16)
points((train[w2,10])~train[w2,11], col="red", pch=16)
points((train[c(w4,w5),10])~train[c(w4,w5),11], col="red", pch=16)
plot((train[,10])~train[,12], main="Median Housing v Near Ocean", xlab="Near Ocean", ylab="log(median house value)")
points((train[w1,10])~train[w1,12], col="red", pch=16)
points((train[w2,10])~train[w2,12], col="red", pch=16)
points((train[c(w4,w5),10])~train[c(w4,w5),12], col="red", pch=16)
plot((train[,10])~train[,13], main="Median Housing v Oneh_Ocean", xlab="One hour drive from ocean", ylab="log(median house value)")
points((train[w1,10])~train[w1,13], col="red", pch=16)
points((train[w2,10])~train[w2,13], col="red", pch=16)
points((train[c(w4,w5),10])~train[c(w4,w5),13], col="red", pch=16)
plot((train[,10])~train[,7], main="Median Housing v Population", xlab="Population", ylab="log(median house value)")
points((train[w1,10])~train[w1,7], col="red", pch=16)
points((train[w2,10])~train[w2,7], col="red", pch=16)
points((train[c(w4,w5),10])~train[c(w4,w5),7], col="red", pch=16)
plot((train[,10])~train[,8], main="Median Housing v households", xlab="Households", ylab="log(median house value)")
points((train[w1,10])~train[w1,8], col="red", pch=16)
points((train[w2,10])~train[w2,8], col="red", pch=16)
points((train[c(w4,w5),10])~train[c(w4,w5),8], col="red", pch=16)
plot((train[,10])~train[,4], main="Median Housing v Housing Median Age", xlab="Housing Median Age", ylab="log(median house value)")
points((train[w1,10])~train[w1,4], col="red", pch=16)
points((train[w2,10])~train[w2,4], col="red", pch=16)
points((train[c(w4,w5),10])~train[c(w4,w5),4], col="red", pch=16)
#so many
##########################################################################
####### SELECTING VARIABLES VIA STEPWISE SELECTION ############
#########################################################################

#remove predictors with stepwise 
library(MASS)
stepAIC(lm((median_house_value) ~ ., data=train[,-c(1,14,6,5,3)]), direction = "both", k=2)

# so the model stepwise selection picks has only housing median age, total rooms, population, median_income
#near bay, near ocean, on ocean, only got rid of longitude 

#look at stepwise model 
mod_step <- lm(median_house_value ~ ., data=train[,-c(1,14,6,5,3,2)]) 
summary(mod_step) #all sig. predictors , R squared is bigger than full 3
AIC(mod_step)
BIC(mod_step)
AICc(mod_step)
summary(full3)  #compare models 
vif(mod_step) #GOOD

# check assumptions of stepwise model 
par(mfrow=c(3,4))
plot(rstandard(mod_step)~fitted(mod_step), xlab="fitted", ylab="Residuals")
for(i in c(4,7:13)){
  plot(rstandard(mod_step)~train[,i], xlab=names(train)[i], ylab="Residuals")
}
qqnorm(rstandard(mod_step))
qqline(rstandard(mod_step))
plot((train$median_house_value)~fitted(mod_step))
abline(a=0,b=1)
lines(lowess((train$median_house_value)~fitted(mod_step)), col="blue")
#looks good, no patterns, normality is a lil wiggly 
# what about if we tried all subsets?


##########################################################################
####### SELECTING VARIABLES VIA ALL POSSIBLE SUBSETS METHOD ############
#########################################################################
install.packages("leaps")
library(leaps)
best <- regsubsets((median_house_value)~., data=train[,-c(1,14,6,5,3)], nbest=1) #dont use response, or colinear predictors  
#output one model for rach # predictors
summary(best) 
# let's plot these for easier digestibility
subsets(best, statistic="adjr2", main = "Adjusted R Squared Values for Each Subset Model")
# so this shows that essentially every model with 4 or more predictors has similar adjusted R-squares
# or even slightly smaller

# let's look at all our summary measures:
select_criteria = function(model, n)
{
  SSres <- sum(model$residuals^2)
  Rsq_adj <- summary(model)$adj.r.squared
  p <- length(model$coefficients) - 1
  AIC <- n*log(SSres/n) + 2*p
  AICc <- AIC + (2*(p+2)*(p+3)/(n-p-1))
  BIC <- n*log(SSres/n) + (p+2)*log(n)
  res <- c(SSres, Rsq_adj, AIC, AICc, BIC)
  names(res) <- c("SSres", "Rsq_adj", "AIC", "AIC_c", "BIC")
  return(res)
}

# so we agree with what we got in stepwise, but 7 or 8 predictor models could work out
m7 <- lm((median_house_value)~., data=train[,-c(1,2,3,5,6,10,14)])
summary(m7)
m8 <- lm((median_house_value)~., data=train[,-c(1,3,5,6,10,14)])
summary(m8)
AIC(m8)
BIC(m8)
AICc(m8)

vif(m7) #same as last
vif(m8) #same as last 
vif(mod_step) #compare to stepwise 

#between mod 7 (step),8 

#Check Model7 assumptions
par(mfrow=c(3,4))
plot(rstandard(m7)~fitted(m7), xlab="fitted", ylab="Residuals", main = "Residual Plot")
for(i in c(9,11,12,13,7,8,4)){
  plot(rstandard(m7)~(train[,i]), xlab=names(train)[i], ylab="Residuals", main = "Residual Plot")
}
qqnorm(rstandard(m7))
qqline(rstandard(m7))
plot((train$median_house_value)~fitted(m7),ylab = "Median House Value", main = "Response vs. Fitted Values")
abline(a=0,b=1)
lines(lowess((train$median_house_value)~fitted(m7)), col="blue")

#Check Model8 assumptions
par(mfrow=c(2,4))
plot(rstandard(m8)~fitted(m8), xlab="fitted", ylab="Residuals")
for(i in c(9,11,12,13,7,8,4)){
  plot(rstandard(m8)~(train[,i]), xlab=names(train)[i], ylab="Residuals")
}
qqnorm(rstandard(m8))
qqline(rstandard(m8))
plot((train$median_house_value)~fitted(m8))
abline(a=0,b=1)
lines(lowess((train$median_house_value)~fitted(m8)), col="blue")
#model assumptions cannot break the tie 

#CHOSEN Mod_step
#has smaller AIC, BIC, and higher R^2 
BIC(m7)
BIC(mod_step)
summary(m7)$adj.r.squared
summary(mod_step)$adj.r.squared

##########################################################################
#################### VALIDATE SELECTED MODEL ############################
#########################################################################

#Compare data in train and test 
summary(train)
summary(test)
#looks comparable, which is good

#transform response 
test$median_house_value <- log(test$median_house_value) #because 0 in CI, and rounded power = 0 

#Create model
modstep_test <- lm((median_house_value)~ housing_median_age + population + households + median_income + near_bay+ near_ocean + oneh_ocean, data=test)
summary(modstep_test) #smaller adjusted R2, significant linear relationship (pvalue)
summary(modstep_test)
#similar R^2, both significant, all significant, similar estimates 

# check that the assumptions still hold
pairs(test[,-1]) 
pairs(test[,c(4,7,8,9,11,12,13)]) # still similar median income and housing median age is lifted in the test

par(mfrow=c(3,3))
plot((test$median_house_value)~fitted(modstep_test))
abline(a=0,b=1)
lines(lowess((test$median_house_value)~fitted(modstep_test)), col="blue")
plot(rstandard(modstep_test)~fitted(modstep_test), xlab="fitted", ylab="Residuals")
#one very obvious outlier but everything else looks the same 
for(i in c(4,7,8,9,11,12,13)){
  plot(rstandard(modstep_test)~test[,i], xlab=names(test)[i], ylab="Residuals")
}
qqnorm(rstandard(modstep_test))
qqline(rstandard(modstep_test))
#normality looks a little worse

##########################################################################
########## LEVERAGE, OUTLIERS, AND INFLUENTIAL POINTS ###################
#########################################################################
#leverage
h <- hatvalues(modstep_test)
hcut <- 2*length(coefficients(modstep_test))/nrow(test)
w1 <- which(h > hcut)
w1
#so many, same ones 

#outliers
r <- rstandard(modstep_test)
w2 <- which(r >=2 | r <= -2)
w2
#so many still 

# check for influence
d <- cooks.distance(modstep_test)
dcut <- qf(0.5, length(coef(modstep_test)), modstep_test$df.residual) #use f distribution
w3 <- which(d > dcut)
w3
#none, same as last 

#influence your own fitted value 
dffit <- dffits(modstep_test)
cutfit <- 2*sqrt(length(coef(modstep_test))/nrow(test))
w4 <- which(abs(dffit) > cutfit)
w4
#so many still 

dfbeta <- dfbetas(modstep_test)
cutb <- 2/sqrt(nrow(test))
w5 <- which(abs(dfbeta[,1]) > cutb | abs(dfbeta[,2]) > cutb | abs(dfbeta[,3]) > cutb | abs(dfbeta[,4]) > cutb| abs(dfbeta[,5]) > cutb| abs(dfbeta[,6]) > cutb)
w5



