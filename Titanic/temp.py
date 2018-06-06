import csv, re
import numpy as np
from keras import Sequential
from keras.layers import Dense

# Age patterns
p1 = re.compile(r"0\.")
# Ticket patterns
p3 = re.compile(r"(\d*?)$")

def getAge(arg):
    fract = p1.match(arg)
    if arg == "":
        return 32
    elif arg[:2] == "0.":
        return int(arg[fract.span()[1]:])
    elif arg[-2:] == ".5":
        return int(arg[:-2])
    else:
        return int(arg)

def getTicket(arg):
    search = p3.search(arg)
    ticket = arg[search.span()[0]:]
    if ticket == "":
        return 0
    else:
        return int(ticket)

def dataclean(inputarray):
    #PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    output = []
    v = 1
    embarkedMapping = {"S":1, "Q":2, "C":3, "":1}
    # skipping the first row
    for row in inputarray:
        if row[0] == "PassengerId":
            pass
        else:
            if len(row) == 12:
                v = 2
            pclass = int(row[v].strip())
            name = row[v+1].strip()
            sex = 0 if row[v+2].strip() == "male" else 1
            age = getAge(row[v+3].strip())
            x = row[v+4].strip()
            sibsp = int(x) if x in ['1', '0', '3', '4', '2', '5', '8'] else 0
            x = row[v+5].strip()
            parch = int(x) if x in ['0', '1', '2', '5', '3', '4', '6'] else 0
            ticket = getTicket(row[v+6].strip())
            fare = float(row[v+7].strip())
            # Cabin - skip
            cabin = row[v+8].strip()
            embarked = embarkedMapping[row[v+9].strip()]

            output += [[pclass, sex, age, sibsp, parch, fare, embarked]]
    return output

def getlabels(arg):
    labels = []
    for row in arg:
        if row[0] == "PassengerId":
            pass
        else:
            labels += [int(row[1])]
    return labels

with open("train.csv", mode='r') as f:
    x = list(csv.reader(f))
    x_train = dataclean(x)
    y_train = getlabels(x)

    x_test = np.array(x_train[-100:])
    y_test = np.array(y_train[-100:])

    x_train = np.array(x_train[:-100])
    y_train = np.array(y_train[:-100])

    model = Sequential()
    model.add(Dense(7, input_shape=(7,)))
    model.add(Dense(3))
    model.add(Dense(1, activation="softmax"))

    model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=5)

    score = model.evaluate(x_test, y_test)

    print(score)
