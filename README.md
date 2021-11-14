## Machine Learning DevOps Engineer Nanodegree Program
# Project: Deploying a Machine Learning Model on Heroku with FastAPI

Github url: https://github.com/ainfinum/mlops-project3
 
 
## Model

* Random forest classifier trained to predict whether a person makes over 50K a year on Census Income Data Set

## API endpoints

### Request & Response Examples


### GET

* Hello page
   * https://mlops-project3.herokuapp.com/

* API documentation
   * https://mlops-project3.herokuapp.com/docs


### POST
* Predict endpoint
   * https://mlops-project3.herokuapp.com/predict

   Request body:

```
{
    "age": 40,
    "workclass": "State-gov",
    "fnlgt": 54545,
    "education": "Bachelors",
    "educationNum": 13,
    "maritalStatus": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capitalGain": 2174,
    "capitalLoss": 0,
    "hoursPerWeek": 40,
    "nativeCountry": "United-States"
}
```
   Response body:

```
{'salary': '<=50K'}
```
