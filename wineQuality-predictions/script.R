install.packages("ggplot2")
install.packages("ggcorrplot")
install.packages("caret")
install.packages("e1071")
install.packages("factoextra")
install.packages("randomForest")
install.packages("rpart")
install.packages("ROCR")
install.packages("pROC")
install.packages("ROSE")
install.packages("kernlab")
install.packages("party")
install.packages("MASS")

library(e1071)
#funzione per eseguire training e test dell'SVM
compute.svm = function(trainset, testset, kernel, cost=1, gamma=1, mode="sens_spec", positive = "Alta" ){
  svm.model = svm(quality_label ~ .,
                  data = trainset,
                  type = "C-classification",
                  kernel = kernel,
                  gamma = gamma,
                  cost = cost)
  
  #testing del mdodello
  prediction.svm = predict(svm.model, testset)
  svm.table = table(testset$quality_label, prediction.svm)
  confusionMatrix(svm.table, mode = mode, positive = positive)
}

library(randomForest)
#funzione per eseguire training e test con Random Forest
compute.randomForest = function(trainset, testset, mode="sens_spec", positive = "Alta"){
  
  forest.model = randomForest(quality_label ~ ., data=trainset)
  prediction.forest = predict(forest.model, newdata = testset)
  forest.table = table(prediction.forest, testset$quality_label)
  confusionMatrix(forest.table, mode = mode, positive = positive)
}

#caricamento del dataaset
wine = read.csv("winequality-white.csv", header = TRUE, sep = ";")

str(wine)
wine.data = wine
wine.data$quality_label = "Media"
wine.data$quality_label[wine.data$quality < 6] = "Bassa"
wine.data$quality_label[wine.data$quality > 7] = "Alta"
wine.data$quality_label = factor(wine.data$quality_label)

# seleziono tutte le variabili tranne "quality"
wine.active = wine.data[,-c(12)]
sum(is.na(wine.active)) # non ci sono valori nulli

#DISTRIBUZIONE CLASSI
table.distribuzione = table(wine.data$quality)
barplot(table.distribuzione)

table.distribuzione = table(wine.active$quality)
barplot(table.distribuzione)

# ANALISI UNIVARIATA
# è importante dividere il dataset in attributi input e target
x = wine.active[, 1:11]
y = wine.active[, 12]

# Plot degli outliers
oldpar = par(mfrow = c(2, 6))
for (i in 1:11) {
  boxplot(x[, i], main = names(wine.active)[i])
}


library(caret)
# Plot distribuzione attributi per qualità
featurePlot(
  x,
  y,
  plot = "density",
  scales = list(
    x = list(relation = "free"),
    y = list(relation = "free")
  ),
  auto.key = list(columns = 3)
)

# Distribuzione dei valori predittori

library("MASS")
par(mar=c(1,1,1,1))
oldpar = par(mfrow = c(3,2))
for ( i in 1:6 ) {
  truehist(wine.active[[i]], xlab = names(wine.active)[i], col = 'lightgreen', main = paste("Average =", signif(mean(wine.active[[i]]),3)), nbins = 50)
}
for ( i in 7:11) {
  truehist(wine.active[[i]], xlab = names(wine.active)[i], col = 'lightgreen', main = paste("Average =", signif(mean(wine.active[[i]]),3)), nbins = 50)
}

### ANALISI MULTIVARIATA ###

# Matrice di correlazione
library(ggcorrplot)
ggcorrplot(
  cor(wine),
  hc.order = TRUE,
  type = "lower",
  lab = TRUE,
  insig = "blank"
)


# Boxplot per relazione tra volume di alcol e qualità
ggplot(data = wine.active, aes(y=alcohol)) + geom_boxplot(aes(fill=quality_label))

# Determinamo se c'è correlazione lineare tra "total.sulfur.dioxide" - "density" - "sulphates" - "alcohol"
pairs(wine.active[, c(7, 8, 10, 11)],
      col = wine.active$quality_label,
      oma = c(3, 3, 3, 15))
par(xpd = TRUE)
legend(
  "bottomright",
  fill = unique(wine.active$quality_label),
  legend = c(levels(wine.active$quality_label))
)

# Plot relazione densità e alcohol con qualità 
ggplot(wine.active, aes(
  x = density ,
  y = alcohol,
  colour = factor(quality_label)
))  +
  geom_point() +
  labs(x = "density",
       y = "alchool",
       title = "Relazione tra densità e alcohol e la loro classificazione") +
  theme_minimal()

# Plot relazione residual.sugar e densità con qualità 
ggplot(wine.active, aes(
  x = density,
  y = residual.sugar,
  colour = factor(quality_label)
))  +
  geom_point() +
  labs(x = "density",
       y = "residual.sugar",
       title = "Relazione tra residual.sugar e density e la loro classificazione") +
  theme_minimal()



### PRIMO MODELLO SVM
#creo training e test set
ind = sample(2,
             nrow(wine.active),
             replace = TRUE,
             prob = c(0.7, 0.3))
testset = wine.active[ind == 2, ]
trainset = wine.active[ind == 1, ]

#train del modello senza parametri costo e gamma
compute.svm(trainset, testset, "radial")

#tune model per determinare paramentri migliori
tune.out = tune(
  svm,
  quality_label ~ .,
  data = trainset,
  kernel = "radial",
  ranges = list(
    cost = c(0.1 , 1 , 10 , 100 , 1000),
    gamma = c(0.5, 1, 2, 3, 4)
  )
)

summary(tune.out) # best parameters: cost=1, gamma=0.5


#train del modello con parametri cost=10, gamma=0.5
compute.svm(trainset, testset, "radial", cost=1, gamma=0.5)


#################################################################

# divido il dataset in due classi (alta qualità, bassa qualità )
wine_ridotto = wine

wine_ridotto$quality_label = "Alta"
wine_ridotto$quality_label[wine_ridotto$quality < 7] = "Bassa"
wine_ridotto$quality_label = factor(wine_ridotto$quality_label)
wine_ridotto.active = wine_ridotto[, -c(12)]

table.distribuzione = table(wine_ridotto.active$quality_label)
barplot(table.distribuzione)

#creo training e test set
ind = sample(2,
             nrow(wine_ridotto.active),
             replace = TRUE,
             prob = c(0.7, 0.3))
testset.wine_ridotto = wine_ridotto.active[ind == 2, ]
trainset.wine_ridotto = wine_ridotto.active[ind == 1, ]


compute.svm(trainset.wine_ridotto, testset.wine_ridotto, "radial")

#tune model per determinare paramentri migliori
tune.out = tune(
  svm,
  quality_label ~ .,
  data = trainset.wine_ridotto,
  kernel = "radial",
  ranges = list(
    cost = c(0.1 , 1 , 10 , 100 , 1000),
    gamma = c(0.5, 1, 2, 3, 4)
  )
)

summary(tune.out) #best costo: 10, gamma: 1

compute.svm(trainset.wine_ridotto, testset.wine_ridotto, "radial", cost=10, gamma=1)


###############################################
## SECONDO MODELLO: RANDOM FOREST
compute.randomForest(trainset, testset)


#### testo su dataset con classe ALTA e BASSA
compute.randomForest(trainset.wine_ridotto, testset.wine_ridotto)

#########################################
#precision, recall, fmeasure dei modelli

# SVM dataset ridotto (Alta/Bassa)
compute.svm(trainset.wine_ridotto, testset.wine_ridotto, "radial", cost=10, 
            gamma=1, mode = "prec_recall", positive = "Bassa") 

compute.svm(trainset.wine_ridotto, testset.wine_ridotto, "radial", cost=10, 
            gamma=1, mode = "prec_recall", positive = "Alta")

# Random Forest dataset ridotto (Alta/Bassa)
compute.randomForest(trainset.wine_ridotto, testset.wine_ridotto, 
                     mode = "prec_recall", positive = "Alta") 

compute.randomForest(trainset.wine_ridotto, testset.wine_ridotto, 
                     mode = "prec_recall", positive = "Bassa")

################################################################
###   CURVA ROC ###
library(ROSE)

# Dataset qualità Alta/Bassa
svm.fit = svm(quality_label ~ .,
              data = trainset.wine_ridotto,
              type = "C-classification",
              kernel = "radial",
              gamma = 10,
              cost = 1,
              prob = TRUE)

# Predizione con SVM usando Dataset qualità Alta/Bassa
pred = predict(svm.fit, testset.wine_ridotto, prob = TRUE)

# curva ROC SVM
roc.curve(testset.wine_ridotto$quality_label, pred)


# Dataset qualità Alta/Bassa
wine.rf = randomForest(quality_label ~ ., data=trainset.wine_ridotto)

# Predizione con RF usando Dataset qualità Alta/Bassa
y_pred = prediction.forest = predict(wine.rf, newdata = testset.wine_ridotto)

# curva ROC RF
roc.curve(testset.wine_ridotto$quality_label, y_pred)

############### Model comparison ##############
# 10-fold cross-validation
library(pROC) 
library(kernlab)
library(party)

# Dataset qualità alta/bassa
control = trainControl(method = "repeatedcv", number = 10, repeats = 3,
                       classProbs = TRUE, summaryFunction = twoClassSummary)

# train SVM 10-fold
svm.model = train(quality_label ~ .,
                  data = trainset.wine_ridotto,
                  gamma = 1,
                  cost = 10,
                  method = "svmRadial", metric = "ROC", trControl = control)
# train RF 10-fold
randomforest.model = train(quality_label ~ .,
                           data = trainset.wine_ridotto,
                           method = "rf", 
                           metric = "ROC",
                           trControl = control)

# predizione SVM
svm.probs = predict(svm.model, testset.wine_ridotto[,! names(testset.wine_ridotto) %in% c("quality_label")],
                    type = "prob")

# predizione RF
rf.probs = predict(randomforest.model, testset.wine_ridotto[,! names(testset.wine_ridotto) %in% c("quality_label")],
                   type = "prob")

# curva ROC SVM dopo 10-fold
svm.ROC = roc(response = testset.wine_ridotto$quality_label, predictor = svm.probs$Bassa,
              levels = levels(testset.wine_ridotto$quality_label))
plot(svm.ROC, type = "S", col = "red")

# curva ROC RF dopo 10-fold
rf.ROC = roc(response = testset.wine_ridotto$quality_label, predictor = rf.probs$Bassa,
             levels = levels(testset.wine_ridotto$quality_label))
plot(rf.ROC, add = TRUE, col = "blue")

svm.ROC
rf.ROC

# vari plot di confronto tra SVM e RF con valori AUC e ROC 
cv.values <- resamples(list(rf = randomforest.model, svm = svm.model))
summary(cv.values)
dotplot(cv.values, metric = "ROC") 

bwplot(cv.values, layout = c(3, 1))
splom(cv.values,metric="ROC")

# prestazioni temporali SVM vs RF
cv.values$timings
