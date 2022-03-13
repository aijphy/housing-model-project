As I am learning about MLOps, I decided to practice training a model and deploying a decision tree regressor to predict housing prices. The model follows the project https://towardsdatascience.com/machine-learning-project-predicting-boston-house-prices-with-regression-b4e47493633d.  The data is from 1978, and data drift is expected beyond inflation which was accounted for. The purpose of this exercise was not to have a useful model, but to practice setting up, training, and deploying a model.

The model was trained locally, and uploaded as a pickle file to an S3 bucket. A lambda function (lambda_function.py) was created which could be triggered by a REST API, using a GET request with parameters included in the header. Using GET requests is not the best for security purposes, but was used as an easy way of testing. In order to use the scikit-learn modules, a lambda layer was added from https://github.com/model-zoo/scikit-learn-lambda/blob/master/layers.csv.

- Alex Johnson
