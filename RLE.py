# based on http://rosettacode.org/wiki/Run-length_encoding#Python
from bitstring import BitArray, BitStream

def encode(input):
	count = 1
	prev = input[0]
	lst = []
	for bit in input[1:]:
		if bit != prev:
			entry = (prev,count)
			lst.append(entry)
			count = 1
			prev = bit
		else:
			count += 1
	else:
		entry = (bit,count)
		lst.append(entry)
	return lst
 
 
def decode(lst):
	q = BitStream()
	temp = BitArray('0b0')
	for bit, count in lst:
		for i in xrange(count):
			temp.bool = bit
			q.append(temp)
	return q