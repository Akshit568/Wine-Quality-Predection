
# Load libraries
library(caret)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(reshape2)
library(class)  # For KNN
library(pROC)   # For ROC curve

# Load the dataset
wine <- read.csv("C:/Users/hp/Desktop/Data Science/winequality-red.csv", sep = ";")

# Convert the quality column to a binary classification (Good/Bad)
wine$quality <- factor(ifelse(wine$quality >= 6, "Good", "Bad"))

# Compute the correlation matrix (excluding the 'quality' column)
correlation_matrix <- cor(wine[, -ncol(wine)])  # Exclude the 'quality' column

# Melt the correlation matrix for ggplot2
correlation_matrix_melted <- melt(correlation_matrix)

# Plot the heatmap of feature correlations
ggplot(data = correlation_matrix_melted, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  theme_minimal() +
  labs(title = "Heatmap of Feature Correlations", x = "", y = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Split the data into training and testing sets
set.seed(123)  # For reproducibility
index <- createDataPartition(wine$quality, p = 0.7, list = FALSE)
train_data <- wine[index, ]
test_data <- wine[-index, ]

# Normalize the features
preproc <- preProcess(train_data[, -ncol(train_data)], method = c("center", "scale"))
train_data_scaled <- predict(preproc, train_data)
test_data_scaled <- predict(preproc, test_data)

# Train the Decision Tree model
set.seed(123)
tree_model <- train(
  quality ~ ., 
  data = train_data_scaled, 
  method = "rpart",
  trControl = trainControl(method = "cv", number = 5)  # 5-fold cross-validation
)

# Train the KNN model using class library
# In KNN, the training data must be supplied as a matrix of predictors and a factor response variable
set.seed(123)
knn_model <- knn(
  train = train_data_scaled[, -ncol(train_data)],  # Predictor variables
  test = test_data_scaled[, -ncol(test_data)],    # Test predictors
  cl = train_data_scaled$quality,                  # Response variable (quality)
  k = 5                                             # Number of neighbors
)

# View the details of both models
print("Decision Tree Model:")
print(tree_model)
cat("\n")

# Plot the decision tree with color
rpart.plot(tree_model$finalModel, 
           type = 3,            # Type of plot: splits as circles, terminal nodes as rectangles
           extra = 102,         # Show information in the nodes
           box.palette = "RdBu",  # Red-Blue palette for node colors
           branch.lty = 3,      # Dashed branches for better visibility
           branch.col = "blue", # Color branches
           split.box.col = "white", # Color for split boxes
           shadow.col = "gray",  # Shadow for better aesthetics
           main = "Decision Tree: Wine Quality (Good/Bad)")  # Title for the plot

# Make predictions on the test set for Decision Tree
tree_predictions <- predict(tree_model, test_data_scaled)
tree_conf_matrix <- confusionMatrix(tree_predictions, test_data_scaled$quality)
cat("Decision Tree Model Accuracy: ", round(tree_conf_matrix$overall['Accuracy'] * 100, 2), "%\n")

# Evaluate KNN accuracy (KNN predictions already generated)
knn_conf_matrix <- confusionMatrix(knn_model, test_data_scaled$quality)
cat("KNN Model Accuracy: ", round(knn_conf_matrix$overall['Accuracy'] * 100, 2), "%\n")

# Plot confusion matrices for both models

# Decision Tree Confusion Matrix Heatmap
tree_conf_matrix_data <- as.data.frame(tree_conf_matrix$table)
ggplot(data = tree_conf_matrix_data, aes(Prediction, Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "Confusion Matrix for Decision Tree", x = "Predicted", y = "Actual") +
  theme_minimal()

# KNN Confusion Matrix Heatmap
knn_conf_matrix_data <- as.data.frame(knn_conf_matrix$table)
ggplot(data = knn_conf_matrix_data, aes(Prediction, Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "Confusion Matrix for KNN", x = "Predicted", y = "Actual") +
  theme_minimal()

# ROC Curve for KNN Model

# First, convert predictions to numeric
knn_probabilities <- as.numeric(knn_model)  # KNN returns factors, so convert to numeric
knn_true_values <- as.numeric(test_data_scaled$quality)  # Actual values as numeric

# Create ROC curve using pROC
roc_knn <- roc(knn_true_values, knn_probabilities)
plot(roc_knn, main = "ROC Curve for KNN", col = "blue", lwd = 2)

# Display AUC (Area Under Curve)
cat("AUC for KNN Model: ", auc(roc_knn), "\n")
###################################add after tis add oter 
# Load necessary libraries
library(e1071)
library(ggplot2)
library(caret)

# Load the dataset from your specified path
wine_data <- read.csv("C:/Users/hp/Desktop/Data Science/winequality-red.csv", sep=";")

# Convert the 'quality' variable to a factor for classification
wine_data$quality <- as.factor(wine_data$quality)

# Display the structure of the data
str(wine_data)

# 1. Train Naive Bayes model using two features ('alcohol' and 'volatile acidity') for visualization
set.seed(123)  # For reproducibility
nb_model <- naiveBayes(quality ~ alcohol + volatile.acidity, data=wine_data)

# Summary of the Naive Bayes model
print(nb_model)

# 2. Predict on the training dataset
predictions <- predict(nb_model, wine_data)

# 3. Evaluate model performance using confusion matrix
confusion_matrix <- confusionMatrix(predictions, wine_data$quality)
print(confusion_matrix)

# 4. Add predictions to the dataset for visualization
wine_data$predicted_quality <- predictions

# 5. Visualize Naive Bayes results using a scatter plot
ggplot(wine_data, aes(x=alcohol, y=volatile.acidity, color=predicted_quality, shape=quality)) +
  geom_point(size=3, alpha=0.7) +
  ggtitle("Naive Bayes Classification: Alcohol vs Volatile Acidity") +
  theme_minimal() +
  labs(x = "Alcohol", y = "Volatile Acidity", color = "Predicted Quality", shape = "Actual Quality") +
  scale_color_brewer(palette = "Set2") +
  theme(legend.position = "bottom")

#------------------------------------------------------------------------------------------->
# Load necessary libraries
library(e1071)
library(ggplot2)
library(caret)

# Load the dataset from your specified path
#wine_data <- read.csv("C:/Users/hp/Desktop/Data Science/winequality-red.csv", sep=";")

# Convert the 'quality' variable to a factor for classification
wine_data$quality <- as.factor(wine_data$quality)

# Display the structure of the data
str(wine_data)

# 1. Train SVM model using two features ('alcohol' and 'volatile acidity') for visualization
set.seed(123)  # For reproducibility
svm_model <- svm(quality ~ alcohol + volatile.acidity, data=wine_data, kernel="linear")

# Summary of the SVM model
summary(svm_model)

# 2. Predict on the training dataset
predictions <- predict(svm_model, wine_data)

# 3. Evaluate model performance using confusion matrix
confusion_matrix <- confusionMatrix(predictions, wine_data$quality)
print(confusion_matrix)

# 4. Visualize SVM results using a scatter plot
ggplot(wine_data, aes(x=alcohol, y=volatile.acidity, color=quality)) +
  geom_point(size=3) +
  stat_contour(data=wine_data, aes(z=as.numeric(predictions)), breaks=0.5, col="black") +
  ggtitle("SVM Classification: Alcohol vs Volatile Acidity") +
  theme_minimal() +
  labs(x = "Alcohol", y = "Volatile Acidity", color = "Wine Quality")

#--------------------------------------------------------------------------------->
# Load necessary libraries
library(ggplot2)

# Load the dataset from your specified path
wine_data <- read.csv("C:/Users/hp/Desktop/Data Science/winequality-red.csv", sep=";")

# Display first few rows to confirm loading
head(wine_data)

# 1. Simple Linear Regression: Predicting 'quality' based on 'alcohol'
model_linear <- lm(quality ~ alcohol, data=wine_data)
summary(model_linear)

# Plot for Simple Linear Regression
ggplot(wine_data, aes(x=alcohol, y=quality)) +
  geom_point() +
  geom_smooth(method='lm', col='blue') +
  ggtitle("Simple Linear Regression: Quality vs Alcohol") +
  theme_minimal()

# 2. Multiple Regression: Predicting 'quality' using all features
model_multiple <- lm(quality ~ ., data=wine_data)
summary(model_multiple)

# Plot for Multiple Regression Predictions
wine_data$predicted_quality <- predict(model_multiple, newdata=wine_data)
ggplot(wine_data, aes(x=predicted_quality, y=quality)) +
  geom_point() +
  geom_abline(slope=1, intercept=0, col='red') +
  ggtitle("Multiple Regression: Actual vs Predicted Quality") +
  theme_minimal()

# 3. Polynomial Regression: Predicting 'quality' using polynomial terms of 'alcohol'
wine_data$alcohol2 <- wine_data$alcohol^2
model_poly <- lm(quality ~ alcohol + alcohol2, data=wine_data)
summary(model_poly)

# Plot for Polynomial Regression
ggplot(wine_data, aes(x=alcohol, y=quality)) +
  geom_point() +
  stat_smooth(method='lm', formula = y ~ poly(x, 2), col='green') +
  ggtitle("Polynomial Regression: Quality vs Alcohol") +
  theme_minimal()
#-------------------------------------------------------------------------------<
# below this add dashboard
# Load necessary libraries
# Load necessary libraries
library(shiny)
library(shinydashboard)
library(ggplot2)
library(rpart.plot)
library(caret)
library(reshape2)

# Define UI for the Shiny Dashboard
ui <- dashboardPage(
  dashboardHeader(title = "Wine Quality Analysis"),
  
  dashboardSidebar(disable = TRUE),  # Disable the sidebar for a single-page layout
  
  dashboardBody(
    # Organize all plots in a layout with three columns per row
    fluidRow(
      # Row 1: Decision Tree Plot, KNN Confusion Matrix
      column(width = 4,
             box(title = "Decision Tree Plot", status = "primary", solidHeader = TRUE, 
                 width = NULL, plotOutput("treePlot", height = 350))),
      
      column(width = 4,
             box(title = "KNN Confusion Matrix", status = "primary", solidHeader = TRUE, 
                 width = NULL, plotOutput("knnConfMatrix", height = 350))),
      
      column(width = 4,
             box(title = "Correlation Heatmap", status = "primary", solidHeader = TRUE, 
                 width = NULL, plotOutput("heatmapPlot", height = 350)))
    ),
    
    fluidRow(
      # Row 2: Naive Bayes Scatter Plot and SVM Scatter Plot
      column(width = 6,
             box(title = "Naive Bayes: Alcohol vs Volatile Acidity", status = "primary", solidHeader = TRUE, 
                 width = NULL, plotOutput("nbPlot", height = 350))),
      
      column(width = 6,
             box(title = "SVM: Alcohol vs Volatile Acidity", status = "primary", solidHeader = TRUE, 
                 width = NULL, plotOutput("svmPlot", height = 350)))
    ),
    
    fluidRow(
      # Row 3: All Regression Plots
      column(width = 4,
             box(title = "Linear Regression: Quality vs Alcohol", status = "primary", solidHeader = TRUE, 
                 width = NULL, plotOutput("linearPlot", height = 350))),
      
      column(width = 4,
             box(title = "Polynomial Regression: Quality vs Alcohol", status = "primary", solidHeader = TRUE, 
                 width = NULL, plotOutput("polynomialRegressionPlot", height = 350))),
      
      column(width = 4,
             box(title = "Multiple Regression", status = "primary", solidHeader = TRUE, 
                 width = NULL, plotOutput("multipleRegressionPlot", height = 350)))
    )
  )
)

# Define server logic for the Shiny dashboard
server <- function(input, output) {
  
  # Render Decision Tree Plot
  output$treePlot <- renderPlot({
    rpart.plot(tree_model$finalModel, type = 3, extra = 102, box.palette = "RdBu")
  })
  
  # Render KNN Confusion Matrix
  output$knnConfMatrix <- renderPlot({
    knn_conf_matrix_data <- as.data.frame(knn_conf_matrix$table)
    ggplot(knn_conf_matrix_data, aes(Prediction, Reference, fill = Freq)) +
      geom_tile() + geom_text(aes(label = Freq), color = "white") +
      scale_fill_gradient(low = "blue", high = "red") +
      labs(title = "KNN Confusion Matrix")
  })
  
  # Render Naive Bayes Scatter Plot
  output$nbPlot <- renderPlot({
    ggplot(wine_data, aes(x = alcohol, y = volatile.acidity, color = nb_predictions, shape = factor(quality))) +
      geom_point(size = 3, alpha = 0.7) +
      labs(title = "Naive Bayes: Alcohol vs Volatile Acidity")
  })
  
  # Render SVM Scatter Plot
  output$svmPlot <- renderPlot({
    ggplot(wine_data, aes(x = alcohol, y = volatile.acidity, color = factor(quality))) +
      geom_point(size = 3) +
      labs(title = "SVM: Alcohol vs Volatile Acidity")
  })
  
  # Render Linear Regression Plot
  output$linearPlot <- renderPlot({
    ggplot(wine_data, aes(x = alcohol, y = quality)) +
      geom_point() + geom_smooth(method = 'lm', col = 'blue') +
      labs(title = "Linear Regression: Quality vs Alcohol")
  })
  
  # Render Polynomial Regression Plot
  output$polynomialRegressionPlot <- renderPlot({
    ggplot(wine_data, aes(x = alcohol, y = quality)) +
      geom_point() +
      geom_smooth(method = 'lm', formula = y ~ poly(x, 2), col = 'green') +
      labs(title = "Polynomial Regression: Quality vs Alcohol")
  })
  
  # Render Multiple Regression Plot
  output$multipleRegressionPlot <- renderPlot({
    model_multiple <- lm(quality ~ ., data = wine_data)
    wine_data$predicted_quality <- predict(model_multiple, wine_data)
    ggplot(wine_data, aes(x = predicted_quality, y = quality)) +
      geom_point() +
      geom_abline(slope = 1, intercept = 0, color = 'red', linetype = "dashed") +
      labs(title = "Multiple Regression: Predicted vs Actual Quality")
  })
  
  # Render Correlation Heatmap Plot
  output$heatmapPlot <- renderPlot({
    cor_matrix <- cor(wine_data[, sapply(wine_data, is.numeric)], use = "complete.obs")
    cor_melt <- melt(cor_matrix)
    ggplot(cor_melt, aes(Var1, Var2, fill = value)) +
      geom_tile(color = "white") +
      scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
      labs(title = "Correlation Heatmap") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
}

# Run the Shiny app
shinyApp(ui = ui, server = server)
-+-+-+-+++++++++++
