import csv, pickle
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

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
        self.variablesNotUsedYet = [self.Cabin, self.Ticket]
        self.target = self.Survived

    def __repr__(self):
        return str(self.PassengerId) + " : " + self.Name

    def cleanArgs(self, args):
        #First lets turn the PassengerId into an int
        args["PassengerId"] = int(args["PassengerId"])

        #Survived := 1 means survived, 0 did not.
        args["Survived"] = int(args["Survived"])

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

def rowToDict(row):
    data = {}
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
    #with open("test.pkl", "rb") as f:
        # PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
        # 891,        0,       3,    "Doo",male,32,0,    0,    370376, 7.75,  ,  Q
        #people = pickle.load(f)
        #print(people)
        with plt.xkcd():
            fig = plt.figure()
            ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
            ax.bar([0, 1], [0, 100], 0.25)
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.xticks([])
            plt.yticks([])
            ax.set_ylim([-30, 10])

            data = np.ones(100)
            data[70:] -= np.arange(30)

            plt.annotate(
                'THE DAY I REALIZED\nI COULD COOK BACON\nWHENEVER I WANTED',
                xy=(70, 1), arrowprops=dict(arrowstyle='->'), xytext=(15, -10))

            plt.plot(data)

            plt.xlabel('time')
            plt.ylabel('my overall health')
            fig.text(
                0.5, 0.05,
                '"Stove Ownership" from xkcd by Randall Monroe',
                ha='center')
            plt.show()
