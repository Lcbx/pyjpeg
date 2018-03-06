import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='homemade image comparison')
parser.add_argument("original", help='path and name of file' )
parser.add_argument("compared", help='path and name of file' )
args = parser.parse_args()

original =cv2.imread(args.original)
compared = cv2.imread(args.compared)

resultmse = np.sum((original.astype("float") - compared.astype("float")) ** 2) / reduce(lambda x, y: x*y, original.shape)
print resultmse