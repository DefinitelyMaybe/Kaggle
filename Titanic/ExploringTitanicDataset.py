import csv, pickle
#import matplotlib.pyplot as plt
import numpy as np
import keras

from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop

class Person(object):
    """Represents a person on the titanic"""
    def __init__(self, args):
        super()
        # args must be a dictionary
        args = self.cleanArgs(args)

        self.PassengerId = args["PassengerId"]
        self.Survived = args["Survived"]
        self.Pclass = args["Pclass"]
        self.Name = args["Name"]
        self.Sex = args["Sex"]
        self.Age = args["Age"]
        self.SibSp = args["SibSp"]
        self.Parch = args["Parch"]
        self.Ticket = args["Ticket"]
        self.Fare = args["Fare"]
        self.Cabin = args["Cabin"]
        self.Embarked = args["Embarked"]

        self.variables = [
                self.Pclass,
                self.Sex,
                self.Age,
                self.SibSp,
                self.Parch,
                self.Fare,
                self.Embarked
                ]
        self.variablesNotUsedYet = [self.Name, self.Cabin, self.Ticket]
        self.target = self.Survived

    def __repr__(self):
        return str(self.PassengerId) + " : " + self.Name

    def cleanArgs(self, args):
        #First lets turn the PassengerId into an int
        args["PassengerId"] = int(args["PassengerId"])

        #Survived := 1 means survived, 0 did not.
        try:
            args["Survived"] = int(args["Survived"])
        except Exception as e:
            args["Survived"] = None

        # Pclass := {1st, 2nd, 3rd} aka ticket class
        try:
            args["Pclass"] = int(args["Pclass"])
        except Exception as e:
            print("check this: Pclass - " + args["Pclass"])

        # not worrying about name at this moment

        # binary features are set as ints will need to check values
        if args["Sex"].lower() == "female":
            args["Sex"] = 0
        elif args["Sex"].lower() == "male":
            args["Sex"] = 1
        else:
            print("check this: Sex - " + args["Sex"])
            args["Sex"] = 2

        # Age variable needs extra cleaning, will be passed back as int
        args["Age"] = self.cleanAgeData(args["Age"])

        # SibSp := # of siblings / spouses aboard the Titanic
        if len(args["SibSp"]) == 1:
            args["SibSp"] = int(args["SibSp"])
        else:
            print("check this: SibSp - " + args["SibSp"])
            # a default value for the moment
            args["SibSp"] = 0

        # Parch := # of parents / children aboard the Titanic
        if len(args["Parch"]) == 1:
            args["Parch"] = int(args["Parch"])
        else:
            print("check this: Parch - " + args["Parch"])
            # a default value for the moment
            args["Parch"] = 0

        # Ticket := ticket number
        """if len(args["Ticket"]) == 0:
            # a default value for the moment
            print("did this ever happen?")
            args["Ticket"] = 0
        else:
            try:
                args["Ticket"] = int(args["Ticket"])
            except Exception as e:
                print("check this: Ticket - " + args["Ticket"])"""

        # Fare := Passenger fare
        try:
            args["Fare"] = float(args["Fare"])
        except Exception as e:
            print("check this: Fare - " + args["Fare"])
            args["Fare"] = 0.0

        # cabin := Cabin number
        # leaving cabin data for the moment
        """try:
            if len(args["Cabin"]) == 0:
                # a default value for the moment
                args["Cabin"] = 0
            else:
                args["Cabin"] = int(args["Cabin"])
        except Exception as e:
            print("check this: Cabin - " + args["Cabin"])"""

        # embarked := Port of Embarkation {C = 0 = Cherbourg, Q = 1 = Queenstown, S = 2 = Southampton}
        if args["Embarked"] == "C":
            args["Embarked"] = 0
        elif args["Embarked"] == "Q":
            args["Embarked"] = 1
        elif args["Embarked"] == "S":
            args["Embarked"] = 2
        else:
            # a drop box value
            args["Embarked"] = 3

        return args

    def getRowInt(self, attr):
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

    def cleanAgeData(self, arg):
        try:
            if len(arg) == 4:
                if (arg[1]) == ".":
                    arg = int(arg[2:])
                else:
                    arg = int(arg[:-2])
            elif len(arg) == 3:
                arg = int(arg[1:])
            # An unknown age will be given a value of 0
            elif len(arg) == 0:
                arg = 0
            else:
                arg = int(arg)
        except Exception as e:
            print("check this: Age - " + str(arg))
            arg = 0

        return arg

def csvFileToArray(filename):
    def rowToDict(row):
        data = {}
        if len(row) == 12:
            data["PassengerId"] = row[0]
            data["Survived"] = row[1]
            data["Pclass"] = row[2]
            data["Name"] = row[3]
            data["Sex"] = row[4]
            data["Age"] = row[5]
            data["SibSp"] = row[6]
            data["Parch"] = row[7]
            data["Ticket"] = row[8]
            data["Fare"] = row[9]
            data["Cabin"] = row[10]
            data["Embarked"] = row[11]
        else:
            data["PassengerId"] = row[0]
            data["Pclass"] = row[1]
            data["Name"] = row[2]
            data["Sex"] = row[3]
            data["Age"] = row[4]
            data["SibSp"] = row[5]
            data["Parch"] = row[6]
            data["Ticket"] = row[7]
            data["Fare"] = row[8]
            data["Cabin"] = row[9]
            data["Embarked"] = row[10]
        return data

    with open(filename, "r") as f:
        datarows = csv.reader(f)
        data = []
        skip = True
        for row in datarows:
            if skip:
                skip = False
            else:
                data += [Person(rowToDict(row))]
        return data

def modelFromPKL(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def histogramOfAges():
    with open("test.pkl", "rb") as f:
        # PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
        # 891,        0,       3,    "Doo",male,32,0,    0,    370376, 7.75,  ,  Q

        # 0 is False and means they did not survive
        people = pickle.load(f)

        survivors = [p for p in people if p.Survived]
        nonSurvivors = [p for p in people if not p.Survived]


        sAgesData = [p.Age for p in survivors]
        sAgesData = np.array(sAgesData)
        sAgesData = sAgesData[sAgesData.nonzero()]

        nsAgesData = [p.Age for p in nonSurvivors]
        nsAgesData = np.array(nsAgesData)
        nsAgesData = nsAgesData[nsAgesData.nonzero()]

        with plt.xkcd():
            fig, ax = plt.subplots()

            n, bins, patches = ax.hist([sAgesData,nsAgesData], 100, stacked=True, cumulative=True)

            #fig.tight_layout()

            plt.xlabel('age')
            plt.ylabel('count')
            plt.show()

if __name__ == '__main__':
    x = csvFileToArray("test.csv")
    c = 892
    x = np.array([i.variables for i in x])
    #print(x)
    decisionTree = modelFromPKL("titanicDecisionTreeModel-1.pkl")

    with open("submission-1.csv", "w") as f:
        csvFile = csv.writer(f)
        csvFile.writerow(["PassengerId","Survived"])
        for p in x:
            y = decisionTree.predict([p])
            csvFile.writerow([c,y[0]])
            c += 1
