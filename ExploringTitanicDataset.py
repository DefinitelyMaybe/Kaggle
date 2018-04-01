import csv, pickle
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

class Person(object):
    #Not used at this point, left it in just in case
    """Represents a person on the titanic"""
    def __init__(self, args):
        super()
        # args must be a dictionary
        self.id = args["id"]
        self.survived = args["surv"]
        self.pclass = args["pclass"]
        self.name = args["name"]
        self.sex = args["sex"]
        self.age = args["age"]
        self.sibSp = args["sibSp"]
        self.parch = args["parch"]
        self.ticket = args["ticket"]
        self.fare = args["fare"]
        self.cabin = args["cabin"]
        self.embarked = args["embarked"]

    def __repr__(self):
        return self.name

def getRowInt(attr):
    # PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    switch = {
        "PassengerId": 0,
        "Survived":1,
        "Pclass":2,
        "Name":3,
        "Sex":4,
        "Age":5,
        "SibSp":6,
        "Parch":7,
        "Ticket":8,
        "Fare":9,
        "Cabin":10,
        "Embarked":11
    }
    return switch[attr]

def cleanAgeDataforArray(arg):
    if len(arg) == 4:
        if (arg[1]) == ".":
            arg = arg[1:]
        else:
            arg = arg[:-2]
    if len(arg) == 3:
        arg = arg[1:]
    if len(arg) == 0:
        arg = "0"
    return arg

def cleanAgeDataforDict(arg):
    if len(arg) == 1:
        arg = "0{}".format(arg)
    if len(arg) == 4:
        if (arg[1]) == ".":
            arg = arg[1:]
        else:
            arg = arg[:-2]
    if len(arg) == 3:
        arg = arg[1:]
    if len(arg) == 0:
        arg = "0"
    return arg

def valuesFromRowstoDict(rows, value):
    data = {}
    skip = True
    index = getRowInt(value)
    for row in rows:
        #need to ignore the first row of csv data
        if skip:
            if skip:
                skip = False
        else:
            #rules for all the different age strings i.e. 45.5, .83, 1, ' '
            value = row[index]
            if index == 5:
                value = cleanAgeDataforDict(value)

            if value in data:
                data[value] += 1
            else:
                data[value] = 1
    return data

def valuesFromRowstoArray(rows, value, options={}):
    data = []
    skip = True
    index = getRowInt(value)
    for row in rows:
        #need to ignore the first row of csv data
        if skip:
            if skip:
                skip = False
        else:
            #rules for all the different value strings i.e. 45.5, .83, 1, ' '
            value = row[index]
            #index == 5 is Age
            if index == 5:
                value = cleanAgeDataforArray(value)
            data += [int(value)]
    return data

def histogramOfAttr(data, attr):
    # Assumptions made: attr is only ever age.
    d = valuesFromRowstoArray(data, attr)
    x = np.array(d)
    x = x[x.nonzero()]

    fig, ax = plt.subplots()

    n, bins, patches = ax.hist(x, 100)

    ax.set_ylabel("count")
    ax.set_xlabel("age")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    with open("train.csv", "r") as f:
        # PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
        # 891,        0,       3,    "Doo",male,32,0,    0,    370376, 7.75,  ,  Q
        csvData = csv.reader(f)
        #histogramOfAttr(csvData, "Age")
        d = valuesFromRowstoDict(csvData, "Survived")
        for i in d:
            print(i + " : " + str(d[i]))
