
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# storing the data from the csv in a pandas DataFrame
#header is set to zero as the first line of the csv contains the names of the features
#the delimiter argument solves the problem of having a space after each ,
salary_data = pd.read_csv("salary.csv", header = 0, delimiter = ", ")

#some of the features are expressed in strings, the code below fixes this
salary_data["sex"] = salary_data["sex"].apply(lambda row: 0  if row == "Male" else 1)
salary_data["race"] = salary_data["race"].apply(lambda row: 0 if row == "White" else 1)
salary_data["workclass"] = salary_data["workclass"].apply(lambda row: 0 if row == " Without-pay" else 1)
salary_data["occupation"] = salary_data["occupation"].map({"Tech-support":4, 
                                                           "Craft-repair":2,
                                                           "Other-service":2,
                                                           "Sales": 2,
                                                           "Exec-managerial":5, 
                                                           "Prof-specialty":4, 
                                                           "Handlers-cleaners":0, 
                                                           "Machine-op-inspct":2, 
                                                           "Adm-clerical":5, 
                                                           "Farming-fishing":3, 
                                                           "Transport-moving":2, 
                                                           "Priv-house-serv":2, 
                                                           "Protective-serv":3, 
                                                           "Armed-Forces":4})

#the line of code below makes sure there are no NaN values in the dataset
salary_data.fillna(value=2,inplace=True)

#splitting the feauture in data and target
target = salary_data[["income"]]
data = salary_data[["age", "capital-gain", "capital-loss", "hours-per-week", "sex", "education-num", "workclass", "occupation"]]

#splitting the dataset in training and testing data to verify accuracy of the model
training_data, testing_data, training_target, testing_target = train_test_split(data, target, test_size = 0.2, random_state = 1)

#creating the random forest, this forest will have 800 trees
forest = RandomForestClassifier(random_state = 1, n_estimators = 800)

#fitting the forest
forest.fit(training_data, training_target)

#verifying accuracy
accuracy = forest.score(testing_data, testing_target)
print(f"This model predicted salary with {accuracy} accuracy.")

#print(salary_data["education"].value_counts())
