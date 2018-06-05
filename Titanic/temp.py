import csv
import re

# Age patterns
p1 = re.compile(r"0\.")
p2 = re.compile(r".+\.5")
# Ticket patterns
p3 = re.compile(r"$\d*\s")

with open("train.csv", mode='r') as f:
    x = csv.reader(f)
    c = []
    for i in x:
        y = i[8].strip()
        match = p3.match(y)
        if match != None:
            print(y, " : ", match.group(0))

        # id
        # pclass - gets everything with [1, 2, 3]
        # name.. not sure what to do with this one
    """p1 = re.compile(r".*,")
        names = []
        for i in x:
            y = i[2]
            y2 = p1.match(y)
            if y2 != None:
                name = y2.group(0)[:-1]
                if not name in names:
                    names += [name]
        print(names)
        print(len(names))"""
        # sex -["male", "female"] gets everything
        # age - little bit of work but done
    """#Get the fractional ages
        fract = p1.match(y)
        #Get approximate ages
        approx = p2.match(y)
        if y == "":
            pass
        elif fract != None:
            y = y[:fract.span()[0]]
        elif approx != None:
            y = y[approx.span()[1]:]
        elif y == "Age":
            pass
        else:
            y = int(y)"""
        # SibSp with ['1', '0', '3', '4', '2', '5', '8']
        # Parch with ['0', '1', '2', '5', '3', '4', '6']
