
#combinaciones


import itertools

dictt={'a':[1,2,3],'b':['g', 1], 'c':[6]}

varNames = sorted(dictt)
combinations = [dict(zip(varNames, prod)) for prod in itertools.product(*(dictt[varName] for varName in varNames))]

for i in combinations:
    print(i)
#print(combinations)
