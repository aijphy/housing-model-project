# Meant for use in AWS Lambda:
import json
import pickle
import boto3
import sklearn
from sklearn.model_selection import ShuffleSplit, GridSearchCV, train_test_split
from sklearn.metrics import r2_score, make_scorer
from sklearn.tree import DecisionTreeRegressor

print('Loading function')

def lambda_handler(event, context):
      #1. Parse out query string params
      rooms = event['queryStringParameters']['rooms']
      student_teacher_ratio = event['queryStringParameters']['str']
      poverty_percent = event['queryStringParameters']['pov']
      print('rooms, str, pov', rooms, student_teacher_ratio, poverty_percent)


# load model
      s3 = boto3.resource('s3')
      reg = pickle.loads(s3.Bucket("firstmodel1313a").Object("model.pkl").get()['Body'].read())

# fixes an error:   
      setattr(reg,'n_features_',3)

# make prediction:
      cdata = [[rooms, student_teacher_ratio, poverty_percent]]
      price = reg.predict(cdata)[0]

      #2. Construct the body of the response object
      ModelResponse = {}
      ModelResponse['rooms'] = rooms
      ModelResponse['student_teacher_ratio'] = student_teacher_ratio
      ModelResponse['poverty_percent'] = poverty_percent
      ModelResponse['price'] = price
      ModelResponse['message'] = 'Hello from Lambda land'

      #3. Construct http response object
      responseObject = {}
      responseObject['statusCode'] = 200
      responseObject['headers'] = {}
      responseObject['headers']['Content-Type'] = 'application/json'
      responseObject['body'] = json.dumps(ModelResponse)
      print('returning object')
      #4 Return the response object
      return responseObject
