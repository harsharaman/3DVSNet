import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--alpha', nargs=3, type=float, metavar=('a','b','c'),
				help = "alpha modulation for focal loss")#, dest="alpha")

variables = parser.parse_args()				
print(variables.alpha)

#print(parser.alpha)
