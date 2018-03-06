import cv2
import numpy as np
import argparse



####################################
##helper functions
####################################

# the c coefficient used in DCT
def c(x):
	temp = np.sqrt(1./8)
	if(x==0):
		return temp
	else:
		return np.sqrt(2.) * temp

# a matrix mapping the function f(x,y) = x on a 8x8 bloc, transposed it gives f(x,y) = y
matx=np.array([[0.,0.,0.,0.,0.,0.,0.,0.],
					[1.,1.,1.,1.,1.,1.,1.,1.,],
					[2.,2.,2.,2.,2.,2.,2.,2.,],
					[3.,3.,3.,3.,3.,3.,3.,3.,],
					[4.,4.,4.,4.,4.,4.,4.,4.,],
					[5.,5.,5.,5.,5.,5.,5.,5.,],
					[6.,6.,6.,6.,6.,6.,6.,6.,],
					[7.,7.,7.,7.,7.,7.,7.,7.,]])
 
 # the DCT applied to a 8x8 bloc
def DCT(pixels):
	pixels = pixels.astype('int16')
	pixels = pixels-128
	result = np.zeros((8,8),dtype='float')
	for u in range(8):
		for v in range(8):
			result[u][v] =c(u) * c(v) * np.sum(pixels * np.cos( np.pi * (2.*matx +1.) * u /16.) * np.cos( np.pi * (2.*matx.T +1.) * v /16.))
	return result

# the reverse operation of DCT
def iDCT(pixels):
	vc = np.vectorize(c)
	result = np.zeros((8,8),dtype='float')
	for u in range(8):
		for v in range(8):
			result[u][v] = np.sum(vc(matx) * vc(matx.T) * pixels * np.cos( np.pi * (2.*u +1.) * matx /16.) * np.cos( np.pi * (2.*v +1.) * matx.T /16.))
	return result

# a quantification matrix
matQ=np.array([[16,11,10,16,24,40,51,61],
					[12,12,14,19,26,48,60,55],
					[14,13,16,24,40,57,69,56],
					[14,17,22,29,51,87,80,62],
					[18,22,37,56,68,109,103,77],
					[24,35,55,64,81,104,113,92],
					[49,64,78,87,103,121,120,101],
					[72,92,95,98,112,100,103,99]])



####################################
## main program
####################################

parser = argparse.ArgumentParser(description='homemade jpeg')
parser.add_argument("-f", '--filename', nargs = '?', default = "../test.png", help='path and name of file' )
parser.add_argument("-d", '--downsize', nargs = '?',  type = float,  default = 1.0, help='downsizing factor',)
parser.add_argument("-s", '--subsampling', nargs = '?',  choices=['4:4:4', '4:2:2', '4:1:1', '4:2:0',],  default = '4:2:0', help='chroma subsampling',)
parser.add_argument("-q", '--quality', nargs = '?',  type = int,  default = 99, help='quality coefficient',)
args = parser.parse_args()


# based on the quality argument, we scale the quantification matrix
quality = args.quality
if(quality < 50):
	quality=round(5000./float(quality))
else:
	quality=200.-2.*quality
matQ = np.floor((quality*matQ)/100. +1).reshape(8,8).astype(np.int16)


# we load the image
image=cv2.imread(args.filename)
factor = args.downsize
image=cv2.resize(image,(0,0),fx=factor,fy=factor)
YCrCb=cv2.cvtColor(image,cv2.COLOR_RGB2YCR_CB)
height, width, channels = image.shape


# the subsampling at work
if args.subsampling == '4:2:2':
	subYCrCb = np.array([ YCrCb[:,:,0], YCrCb[::2,::, 1], YCrCb[::2,::, 2] ])
elif args.subsampling == '4:1:1':
	subYCrCb = np.array([ YCrCb[:,:,0], YCrCb[::4,::, 1], YCrCb[::4,::, 2] ])
elif args.subsampling == '4:2:0':
	subYCrCb = np.array([ YCrCb[:,:,0], YCrCb[::2,::2, 1], YCrCb[::2,::2, 2] ])
else:
	subYCrCb = np.array([ YCrCb[:,:,0], YCrCb[::,::, 1], YCrCb[::,::, 2] ])


# a little trick to speed up experimentation
# since the rest of compression isn't lossy, we already know the result at this point
#result = np.zeros(image.shape, np.uint8)

# a copy is generated to go on with compression
copy = []
# for each channel (Y, Cr, Cb)
for idx, channel in enumerate(subYCrCb):
	# we make a new corresponding subimage
	rows,cols = channel.shape
	compressed = np.zeros(channel.shape, np.int16)
	
	# for each subarray of 8x8 pixels
	for row in range(0,rows,8):
		for col in range(0,cols,8):
			bloc = channel[row:row+8,col:col+8]
			
			# we apply the DCT and quantification matrix
			compressedBloc = np.round(DCT(bloc)/matQ)
			compressed[row:row+8,col:col+8] = compressedBloc
			
			# the result image is also already computed
			#channel[row:row+8,col:col+8] = np.round(iDCT( compressedBloc * matQ))+128
	
	# the copy gets the compresed subimage
	copy.append(compressed)
	
	# the result gets the alredy decompressed subimage
	#result[:,:,idx] = cv2.resize(channel,(height,width))

# we save the result and original for  the sake of comparison
#resultImage=np.zeros(image.shape,dtype=np.uint8)
#resultImage=cv2.cvtColor(result,cv2.COLOR_YCR_CB2RGB)
#cv2.imwrite("../result.png",resultImage)
#cv2.imwrite("../original.png", image)


####################################
## compression and decompression
####################################

# within a flattened 8x8 array, thes are the indices of a zigzag traversal
Zigzag = [0, 1,8, 16,9,2, 3,10,17,24, 32,25,18,11,4, 5,12,19,26,33,40, 48,41,34,27,20,13,6, 7,14,21,28,35,42,49,56,
			57,50,43,36,29,22,15, 23,30,37,44,51,58, 59,52,45,38,31, 39,46,53,60, 31,54,47, 55,62, 63 ]

# we make a list out of the flattened array via zigzag traversal
fileContent = []
for idx, channel in enumerate(copy):
	rows,cols = channel.shape
	for row in range(0,rows,8):
		for col in range(0,cols,8):
			bloc = channel[row:row+8,col:col+8]
			flattenedBloc = bloc.flatten()
			fileContent.extend(flattenedBloc[Zigzag].tolist())

# huffman encoding
import huffman
filechars = "".join(map(lambda x : str(x) + ",", fileContent))
huffRes = huffman.encode(filechars)

# run length encoding
import RLE
rleRes = RLE.encode(huffRes)

# sizes
hsize = len(huffRes.tobytes())
rsize =  int(len(rleRes) *1.5/8)
print "original size = ", reduce(lambda x, y: x*y, image.shape), " bytes, huffman size = ", hsize , "bytes, compressed size = ", rsize, " bytes"


# decoding
decoded = RLE.decode(rleRes)
decoded = eval("[" + huffman.decode(decoded)[:len(filechars)] + "]")



pointer = 0
final = np.zeros(image.shape, np.uint8)
for idx, channel in enumerate(copy):
	rows,cols = channel.shape
	
	# we undo the zigzag traversal
	for row in range(0,rows,8):
		for col in range(0,cols,8):
			
			bloc = np.zeros((64), np.float32)
			bloc[Zigzag] = decoded[pointer:pointer+64]
			pointer += 64
			
			# apply the inverse DCT
			bloc.resize(8,8)
			channel[row:row+8,col:col+8] = iDCT( (bloc ) * matQ )+128
	
	final[:,:,idx] = cv2.resize(channel,(height,width))

finalImage=np.zeros(image.shape,dtype=np.uint8)
finalImage=cv2.cvtColor(final,cv2.COLOR_YCR_CB2RGB)
cv2.imwrite("../" + str(image.shape) + args.subsampling.replace(":", "-") + "Q" + str(args.quality) + "h" + str(hsize) + "r" + str(rsize)  + ".png",finalImage)






