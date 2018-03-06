import string
from bitstring import BitArray, BitStream



def encode(word):
	
	# a dictionnary for all possible symbol
	symbols = {}
	
	# counts the number of occurences each symbol
	for letter in word:
		if letter in symbols:
			symbols[letter] += 1
		else :
			symbols[letter] = 1
	
	
	# here we construct the binary tree with tuples :
	tree = buildTree(symbols)
	
	# we have to return the tree too for decoding later
	# we could encode it better, but its native string representation is ok
	treeString = str(tree)
	treeString = treeString.replace(", ", ",")
	
	# the code we'll return, which first contains the length taken by the tree
	code = BitStream(12)
	code.int = len(treeString)
	
	
	# then we add the tree to the code
	codedTree = BitArray(8 * len(treeString))
	codedTree.bytes = treeString
	code.append(codedTree)
	
	# a recursive traversal of the tree allows us to create a dictionnary of matching symbols and codes for encoding :
	symbols = buildEncodingDict(tree)
	
	# finally we encode the word
	codedWord = encodeWithDict(symbols, word)
	
	code.append(codedWord)
	
	return code

	
	
	
def decode(code):
	
	# at the beginning of the code is a represetation of the encoding tree
	# we find it thanks to the "#" character that ends it
	
	length = code.read("uint:12")
	
	tree = code.read("bytes:" + str(length))
	
	
	# a simple evaluation builds it
	tree = eval(tree)
	
	
	# we cast the remaining data as bits
	code = code.read("bits")
	
	
	
	# all that's left is to interpret the bits with the tree
	word = ""
	node = tree
	# for each bit
	for t in code :
		 # we iterate through the corresponding child node
		 node = node[t]
		 # if its a leaf node
		 if type(node) is str:
			 # we find the corresponding char
			 temp = node
			 # reset our place in the interpreting tree
			 node = tree
			 # add the char to the final word
			 word += temp
		
	
	return word 
	

	
	
	
	
def buildTree(symbols):

	# the dictionnary is casted to a list of tuples (char, occurences)
	pile = symbols.items()
	
	# while we don't have a lone root node
	while len(pile)>1:
		
		# sort symbols by number of occurences
		pile = sorted(pile, key = lambda x: x[1], reverse = True)
		
		# NOTE : the leaves are the symbols themselves while intermediary nodes are tuples
		
		# get the two nodes with the least total occurences
		nodeLeft, valueLeft = pile.pop()
		nodeRight, valueRight = pile.pop()
		
		# the generated parent node
		parent = (nodeLeft, nodeRight)
		
		# we add it to the pile with its associated total value
		pile.append((parent, valueLeft + valueRight))
		
		
	# the last item is the root node (no need for the occurence count now)
	tree, tot = pile.pop()
	
	return tree
	
def buildEncodingDict(tree):
	# the dictionnary we'll return
	symbols = {}
	
	# a recursive function that parses the tree and fills the dictionnary
	def buildDict(symbols, iteratingNode, buildingCode):
		
		# if its a leaf node, we add it and its representation to the dictionnary
		if type(iteratingNode) is str:
			finalCode = BitArray(buildingCode)
			symbols[iteratingNode] = finalCode
		
		# else it's a parent node
		else:
			
			# within the binary tree : 0 means left child, 1 means right child
			RIGHT = '0b1'
			LEFT = '0b0'
			
			# we build the code corresponding to it's left and right child
			codeLeft = BitArray(buildingCode)
			codeLeft.append(LEFT)
			buildDict(symbols, iteratingNode[0b0], codeLeft)
			
			codeRight = BitArray(buildingCode)
			codeRight.append(RIGHT)
			buildDict(symbols, iteratingNode[0b1], codeRight)
	
	# we call the defined function
	buildDict(symbols, tree, BitArray())
	
	return symbols

	
def encodeWithDict(dictionnary, word):
	codedWord = BitArray()
	for letter in word :
		temp = dictionnary[letter]
		codedWord.append(temp)
	return codedWord	