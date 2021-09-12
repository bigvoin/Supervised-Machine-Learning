load("emnist-letters.mat");% loading the mat file
% Predict Classification Using the K-Nearest Neighbour(KNN) Classifier
% Algorithm using train and test data on the images and the labels of the
% loaded mat file and using the fitcknn algorithm on to train the images
% and the labels and then the predicting algorithm takes the trained data and
% compares it to the new test image then the accuracy of this method is
% displayed by dividing the predicted labels into the length of the test
% labels, also this Examine the resubstitution loss, which, by default, is the fraction of misclassifications from the predictions of Mdl. 
% (For non-default cost, weights, or priors, see the loss.).
% And predicts the incorrect trained data.
%The confusionchart represents the correctly classified observations.
knn_train_features = dataset.train.images;
knn_train_labels = dataset.train.labels;
knn_test_labels = dataset.test.labels;
knn_test_features = dataset.test.images;
knnModel = fitcknn(double(knn_train_features), knn_train_labels);
knnPredictions = predict(knnModel, double(knn_test_features));
knnAccuracy = sum(knnPredictions==dataset.test.labels)/length(dataset.test.labels);
knn_result_err = resubLoss(knnModel);
knnchart = confusionchart(knn_test_labels, knnPredictions);

% Predict Classification Using the Naive Bayes Classifier
% Algorithm using train and test data on the images and the labels of the
% loaded mat file and using the fitcnb algorithm on to train the images
% and the labels and then the predicting algorithm takes the trained data and
% compares it to the new test image then the accuracy of this method is
% displayed by dividing the predicted labels into the length of the test
% labels, also this Examine the resubstitution loss, which, by default, is the fraction of misclassifications from the predictions of Mdl. 
% (For the non-default cost, weights, or priors, see a loss.).
% And predicts the incorrect trained data.
%The confusionchart represents the correctly classified observations.
bayes_train_features = dataset.train.images;
bayes_train_labels = dataset.train.labels;
bayes_test_labels = dataset.test.labels;
bayes_test_features = dataset.test.images;
bayesModel = fitcnb(double(bayes_train_features), bayes_train_labels, "DistributionNames", "mn");
bayesPredictions = predict(bayesModel, double(bayes_test_features));
bayesAccuracy = sum(bayesPredictions==dataset.test.labels)/length(dataset.test.labels);
bayes_result_err = resubLoss(bayesModel);
bayeschart = confusionchart(bayes_test_labels, bayesPredictions);

% Predict Classification Using the Decision Tree Classifier
% Algorithm using train and test data on the images and the labels of the
% loaded mat file and using the fitctree algorithm on to train the images
% and the labels and then the predicting algorithm takes the trained data and
% compares it to the new test image then the accuracy of this method is
% displayed by dividing the predicted labels into the length of the test
% labels, also this Examine the resubstitution loss, which, by default, is the fraction of misclassifications from the predictions of Mdl. 
% (For the non-default cost, weights, or priors, see a loss.).
% And predicts the incorrect trained data.
%The confusionchart represents the correctly classified observations.
tree_train_features = dataset.train.images;
tree_train_labels = dataset.train.labels;
tree_test_labels = dataset.test.labels;
tree_test_features = dataset.test.images;
treeModel = fitctree(double(tree_train_features), tree_train_labels);
treePredictions = predict(treeModel, double(tree_test_features));
treeAccuracy = sum(treePredictions==dataset.test.labels)/length(dataset.test.labels);
tree_result_err = resubLoss(treeModel);
treechart = confusionchart(tree_test_labels, treePredictions);