
#combinaciones


import itertools

parametros = {
                'lr_pesos':  [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15],
                'unidad':    ['binaria', 'gaussiana'],
                'tamBatch':  [10],
                'epocasTRN': [100,200,300],
                'epocasFIT': [2000],

            }

varNames = sorted(parametros)
combinations = [dict(zip(varNames, prod)) for prod in itertools.product(*(parametros[varName] for varName in varNames))]

print(type(combinations))
print(type(combinations[0]))
for i in combinations:
    print(i)

print("Cantidad de Combinaciones posibles: ", len(combinations))
tiempo_x_ejecucion = 6
cantidad_dias = len(combinations) * tiempo_x_ejecucion / 24.0
print("Cantidad de dias continuos de ejecucion: ", cantidad_dias)
#print(combinations)
