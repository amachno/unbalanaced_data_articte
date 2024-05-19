install.packages('dplyr')
install.packages('ggplot2')
install.packages('caret')
install.packages('ROSE')

library(dplyr)
library(mice)
library(ggplot2)
library(psych)
library(tidyverse)
library(caret)
library(tictoc)
library(ROSE)
library(pROC)

### Przygotowanie pliku ze zbiorem zmiennych

D1 <- read.csv("C:/Users/iwoda/repos/unbalanaced_data_articte/data/HFCS 1/HFCS - bazy/HFCS_UDB_3_2_ASCII/D1.csv", stringsAsFactors=TRUE)
H1 <- read.csv("C:/Users/iwoda/repos/unbalanaced_data_articte/data/HFCS 1/HFCS - bazy/HFCS_UDB_3_2_ASCII/H1.csv", stringsAsFactors=TRUE)

D1PL <- D1 %>% filter(SA0100 == "PL") 
H1PL <- H1 %>% filter(SA0100 == "PL")

stopazmienna <- D1PL %>% filter(DL1110ai != 0 | DL1120ai != 0)

zmienneD1 <- stopazmienna %>% select(ID, DA2100, DA3001, DH0001, DH0003,
                                     DH0004, DHHTYPE, DHaged65plus, 
                                     DHEDUH1, DHAGEH1, DHchildrendependent, 
                                     DHGENDERH1, 
                                     DHN013, DODSTOTAL)

zmienneH1 <- H1 %>% select(ID, HI0800, HD1800, HB2400, HDZ0310)

zmienne_projekt <- left_join(zmienneD1, zmienneH1, by = "ID")

df <- zmienne_projekt %>% select(DSTI=DODSTOTAL, GEND=DHGENDERH1, AGE=DHAGEH1, 
                                 EDU=DHEDUH1, HHT=DHHTYPE, HHN=DH0001, HHCH=DHN013, 
                                 HHDCH=DHchildrendependent, HH65=DHaged65plus, 
                                 HHPACT=DH0003, HHEMP=DH0004, ATOT=DA3001, FA=DA2100,
                                 ORE=HB2400, LS=HDZ0310, IA=HD1800, AGH=HI0800)

str(df)
summary(df)
md.pattern(df)
sum(duplicated(df))
df1 <- df[-216,]
FF <- cc(df1)

setwd("C:/Users/iwoda/repos/unbalanaced_data_articte/")
save(FF,file='FF.RData')

### Właściwe badanie

load('FF.Rdata')

FF$GEND <- as.factor(FF$GEND)
FF$EDU <- as.factor(FF$EDU)
FF$HHT <- as.factor(FF$HHT)
FF$ORE <- as.factor(FF$ORE)
FF$LS <- as.factor(FF$LS)
FF$IA <- as.factor(FF$IA)
FF$AGH <- as.factor(FF$AGH)

FF$DSTI <- as.factor(ifelse(FF$DSTI >= 0.3, 
                            "fragile", "resistant"))
summary(FF)


################################
#WIZUALIZACJE - plik R-markdown#
################################

### Modele bazowe
set.seed(123)
podzial <- createDataPartition(FF$DSTI,  
                               p = 0.70,         
                               list = FALSE)

uczacy  <- FF[podzial,]
testowy <- FF[-podzial,]


fitControl <- trainControl(method = "cv",
                           number = 5)

### KNN
sasiedzi <- expand.grid(k = 3:15)

tic()
KNN_model <- train(DSTI ~ .,               
                   data = uczacy,                   
                   method = "knn",                  
                   preProc = c("center", "scale"),  
                   trControl = fitControl,
                   tuneGrid = sasiedzi)   
toc()

KNN_model
plot(KNN_model)

KNN_model_pr <- predict(object = KNN_model, testowy)
confusionMatrix(KNN_model_pr,testowy$DSTI)

### RLR
tic()
RLR_model <- train(DSTI ~ ., data = uczacy,                   
                   method = "regLogistic",
                   preProc = c("center", "scale"),
                   trControl = fitControl,
                   verbosity = 0)
toc()

RLR_model
View(RLR_model)
plot(RLR_model)

RLRImp <- varImp(RLR_model, scale = FALSE)
RLRImp
ggplot(RLRImp, mapping = NULL,
       top = dim(RLRImp$importance)[1]-(dim(RLRImp$importance)[1]-18), environment = NULL) +
  xlab("Feature") +
  ylab("Importance")


RLR_model_pr <- predict(object = RLR_model, testowy)
confusionMatrix(RLR_model_pr,testowy$DSTI)


### XGB
tic()
XGB_model <- train(DSTI ~ ., data = uczacy,                   
                   method = "xgbTree",
                   preProc = c("center", "scale"),
                   trControl = fitControl,
                   verbosity = 0)
toc()

XGB_model
View(XGB_model)
plot(XGB_model)

XGBImp <- varImp(XGB_model, scale = FALSE)
XGBImp
ggplot(XGBImp, mapping = NULL,
       top = dim(XGBImp$importance)[1]-(dim(XGBImp$importance)[1]-18), environment = NULL) +
  xlab("Feature") +
  ylab("Importance")


XGB_model_pr <- predict(object = XGB_model, testowy)
confusionMatrix(XGB_model_pr,testowy$DSTI)

### Modele uwzględniające brak zbalansowania zbioru###

### Pakiet ROSE

table(uczacy$DSTI)

data_balanced_over <- ovun.sample(DSTI ~ ., data = uczacy, method = "over", N = 692)$data
table(data_balanced_over$DSTI)

data_balanced_under <- ovun.sample(DSTI ~ ., data = uczacy, method = "under", N = 126, seed = 1)$data
table(data_balanced_under$DSTI)

data_balanced_both <- ovun.sample(DSTI ~ ., data = uczacy, method = "both", p=0.5, N=409, seed = 1)$data
table(data_balanced_both$DSTI)

data_rose <- ROSE(DSTI ~ ., data = uczacy, seed = 1)$data
table(data_rose$DSTI)


fitControl1 <- trainControl(method = "cv",
                            number = 5,
                            classProbs = TRUE,
                            summaryFunction = twoClassSummary)

### data_balanced_over

KNN_model_over <- train(DSTI ~ .,               
                        data = data_balanced_over,                   
                        method = "knn",                  
                        preProc = c("center", "scale"),  
                        trControl = fitControl1,
                        metric = "ROC",
                        tuneGrid = sasiedzi)

RLR_model_over <- train(DSTI ~ ., 
                        data = data_balanced_over,                   
                        method = "regLogistic",
                        preProc = c("center", "scale"),
                        trControl = fitControl1,
                        metric = "ROC",
                        verbosity = 0)

XGB_model_over <- train(DSTI ~ ., data = data_balanced_over,                   
                        method = "xgbTree",
                        preProc = c("center", "scale"),
                        trControl = fitControl1,
                        metric = "ROC",
                        verbosity = 0)


### data_balanced_under

KNN_model_under <- train(DSTI ~ .,               
                         data = data_balanced_under,                   
                         method = "knn",                  
                         preProc = c("center", "scale"),  
                         trControl = fitControl1,
                         metric = "ROC",
                         tuneGrid = sasiedzi)


RLR_model_under <- train(DSTI ~ ., 
                         data = data_balanced_under,                   
                         method = "regLogistic",
                         preProc = c("center", "scale"),
                         trControl = fitControl1,
                         metric = "ROC",
                         verbosity = 0)


XGB_model_under <- train(DSTI ~ ., data = data_balanced_under,                   
                         method = "xgbTree",
                         preProc = c("center", "scale"),
                         trControl = fitControl1,
                         metric = "ROC",
                         verbosity = 0)


### data_balanced_both

KNN_model_both <- train(DSTI ~ .,               
                        data = data_balanced_both,                   
                        method = "knn",                  
                        preProc = c("center", "scale"),  
                        trControl = fitControl1,
                        metric = "ROC",
                        tuneGrid = sasiedzi)

RLR_model_both <- train(DSTI ~ ., 
                        data = data_balanced_both,                   
                        method = "regLogistic",
                        preProc = c("center", "scale"),
                        trControl = fitControl1,
                        metric = "ROC",
                        verbosity = 0)

XGB_model_both <- train(DSTI ~ ., data = data_balanced_both,                   
                        method = "xgbTree",
                        preProc = c("center", "scale"),
                        trControl = fitControl1,
                        metric = "ROC",
                        verbosity = 0)


### data_rose

KNN_model_rose <- train(DSTI ~ .,               
                        data = data_rose,                   
                        method = "knn",                  
                        preProc = c("center", "scale"),  
                        trControl = fitControl1,
                        metric = "ROC",
                        tuneGrid = sasiedzi)


RLR_model_rose <- train(DSTI ~ ., 
                        data = data_rose,                   
                        method = "regLogistic",
                        preProc = c("center", "scale"),
                        trControl = fitControl1,
                        metric = "ROC",
                        verbosity = 0)


XGB_model_rose <- train(DSTI ~ ., data = data_rose,                   
                        method = "xgbTree",
                        preProc = c("center", "scale"),
                        trControl = fitControl1,
                        metric = "ROC",
                        verbosity = 0)


### SMOTE ###

fitControl2 <- trainControl(method = "repeatedcv",
                            number = 10,
                            repeats = 5,
                            summaryFunction = twoClassSummary,
                            classProbs = TRUE)

fitControl2$sampling <- "smote"

KNN_model_smote <- train(DSTI ~ .,
                         data = uczacy,
                         method = "knn",
                         metric = "ROC",
                         trControl = fitControl2)

RLR_model_smote <- train(DSTI ~ .,
                         data = uczacy,
                         method = "regLogistic",
                         verbose = FALSE,
                         metric = "ROC",
                         trControl = fitControl2)

XGB_model_smote <- train(DSTI ~ .,
                         data = uczacy,
                         method = "xgbTree",
                         verbose = FALSE,
                         metric = "ROC",
                         trControl = fitControl2)

### ROC ###

custom_col <- c("black", "darkgreen", "darkgoldenrod4", "orange2", "cyan3", "indianred")

###ROC-KNN

test_roc_KNN <- function(model, data) {
  
  roc(data$DSTI,
      predict(model, data, type = "prob")[, "fragile"])
}

model_list_KNN <- list(original = KNN_model,
                       over = KNN_model_over,
                       under = KNN_model_under,
                       both = KNN_model_both,
                       rose = KNN_model_rose,
                       smote = KNN_model_smote)

model_list_KNN_roc <- model_list_KNN %>%
  map(test_roc_KNN, data = testowy)

model_list_KNN_roc %>%
  map(auc)

results_list_roc_KNN <- list(NA)
num_mod <- 1
for(the_roc in model_list_KNN_roc){
  
  results_list_roc_KNN[[num_mod]] <- 
    data_frame(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(model_list_KNN)[num_mod])
  
  num_mod <- num_mod + 1
  
}
results_df_roc_KNN <- bind_rows(results_list_roc_KNN)


ggplot(aes(x = fpr,  y = tpr, group = model), data = results_df_roc_KNN) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_minimal() +
  labs(x = "False Positive Rate",
       y = "True Positive Rate",
       title = "Krzywe ROC dla modeli KNN")


###ROC-RLR


test_roc_RLR <- function(model, data) {
  
  roc(data$DSTI,
      predict(model, data, type = "prob")[, "fragile"])
}


model_list_RLR <- list(original = RLR_model,
                       over = RLR_model_over,
                       under = RLR_model_under,
                       both = RLR_model_both,
                       rose = RLR_model_rose,
                       smote = RLR_model_smote)
model_list_RLR_roc <- model_list_RLR %>%
  map(test_roc_RLR, data = testowy)

model_list_RLR_roc %>%
  map(auc)


results_list_roc_RLR <- list(NA)
num_mod <- 1
for(the_roc in model_list_RLR_roc){
  
  results_list_roc_RLR[[num_mod]] <- 
    data_frame(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(model_list_RLR)[num_mod])
  
  num_mod <- num_mod + 1
  
}
results_df_roc_RLR <- bind_rows(results_list_roc_RLR)


ggplot(aes(x = fpr,  y = tpr, group = model), data = results_df_roc_RLR) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_minimal() +
  labs(x = "False Positive Rate",
       y = "True Positive Rate",
       title = "Krzywe ROC dla modeli KNN")


###ROC-XGB


test_roc_XGB <- function(model, data) {
  
  roc(data$DSTI,
      predict(model, data, type = "prob")[, "fragile"])
}


model_list_XGB <- list(original = XGB_model,
                       over = XGB_model_over,
                       under = XGB_model_under,
                       both = XGB_model_both,
                       rose = XGB_model_rose,
                       smote = XGB_model_smote)
model_list_XGB_roc <- model_list_XGB %>%
  map(test_roc_XGB, data = testowy)

model_list_XGB_roc %>%
  map(auc)


results_list_roc_XGB <- list(NA)
num_mod <- 1
for(the_roc in model_list_XGB_roc){
  
  results_list_roc_XGB[[num_mod]] <- 
    data_frame(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(model_list_XGB)[num_mod])
  
  num_mod <- num_mod + 1
  
}
results_df_roc_XGB <- bind_rows(results_list_roc_XGB)


ggplot(aes(x = fpr,  y = tpr, group = model), data = results_df_roc_XGB) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_minimal() +
  labs(x = "False Positive Rate",
       y = "True Positive Rate",
       title = "Krzywe ROC dla modeli XGB")


#####PORÓWNANIE MODELI

XGB_model_under

XGBImp_model_under <- varImp(XGB_model_under, scale = FALSE)
XGBImp_model_under

ggplot(XGBImp_model_under, mapping = NULL, 
       top = dim(XGBImp_model_under$importance)[1]-(dim(XGBImp_model_under$importance)[1]-20), environment = NULL) +
  xlab("Feature") +
  ylab("Importance")+
  theme_minimal()



XGB_model_under_pr <- predict(object = XGB_model_under, testowy)
confusionMatrix(XGB_model_under_pr,testowy$DSTI)

###
RLR_model_under


RLRImp_model_under <- varImp(RLR_model_under, scale = FALSE)
RLRImp_model_under

ggplot(RLRImp_model_under, mapping = NULL, 
       top = dim(RLRImp_model_under$importance)[1]-(dim(RLRImp_model_under$importance)[1]-16), environment = NULL) +
  xlab("Feature") +
  ylab("Importance")+
  theme_minimal()



RLR_model_under_pr <- predict(object = RLR_model_under, testowy)
confusionMatrix(RLR_model_under_pr,testowy$DSTI)