#이미지를 읽어서 ndarry 로 바꾸는 방법 
import PIL.Image as pilimg 
import numpy as np 

img = pilimg.open("./images/1.jpg")
print(type(img))

pix = np.array(img) #ndarray로 바뀐다 
print(pix.shape) #컬러이미지라 3차원이 나온다. 

# for i in range(pix.shape[0]):
#     for j in range(pix.shape[1]):
#         for k in range(pix.shape[2]):
#             print("{0:3}".format(pix[i][i][k], end=' '))
#     print()

print(pix)

#형식을 바꾸어서 저장해보자 
img.save("./images/1.bmp")

#(340, 514, 3) => 340 X 514 X 3 
#80 X 80 

