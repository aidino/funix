# OpenCV Basic

## Open Image in a Notebook

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("path/to/image/image")
# Open CV lưu màu của mỗi pixcel theo dạng BGR,
# matplotlib thì hiển thị với đầu vào RGB, do đó cần phải convert
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)

```

![](opencv_resources/00-open-image.png)

```python
img_gray = cv2.imread('path/to/image/image',cv2.IMREAD_GRAYSCALE)
plt.imshow(img_gray,cmap='gray')
```

![](opencv_resources/00-image-gray.png)



### Resize image

```python
img_rgb.shape
# width, height, color channels
(1300, 1950, 3)
img =cv2.resize(img_rgb,(1300,275))
plt.imshow(img)
```

![](opencv_resources/00-resize1.png)

- **By ratio**

```python
w_ratio = 0.5
h_ratio = 0.5
new_img =cv2.resize(img_rgb,(0,0),img,w_ratio,h_ratio)
plt.imshow(new_img)
```

![](opencv_resources/00-resize2.png)



### Flipping Image

```python
# Along central x axis
new_img = cv2.flip(new_img,0)
plt.imshow(new_img)
```

![](opencv_resources/00-flipping1.png)

```python
# Along central y axis
new_img = cv2.flip(new_img,1)
plt.imshow(new_img)
```

![](opencv_resources/00-flipping2.png)

```python
# Along both axis
new_img = cv2.flip(new_img,-1)
plt.imshow(new_img)
```

![](opencv_resources/00-flipping3.png)

### Saving image files

```python
cv2.imwrite('my_new_picture.jpg',new_img)
```

*Keep in mind, the above stored the BGR version of the image.*



## Open Image with openCV

```python
import cv2

img = cv2.imread('../DATA/00-puppy.jpg',cv2.IMREAD_GRAYSCALE)
# Show the image with OpenCV
cv2.imshow('window_name',img)
# Wait for something on keyboard to be pressed to close window.
cv2.waitKey()
```

## Drawing on Images

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import cv2

blank_img = np.zeros(shape=(512,512,3),dtype=np.int16)
```



### Rectangles

* `img`: Image.
* `pt1`: Vertex of the rectangle.
* `pt2`: Vertex of the rectangle opposite to pt1 .
* `color`: Rectangle color or brightness (grayscale image).
* `thickness`: Thickness of lines that make up the rectangle. Negative values, like #FILLED,mean that the function has to draw a filled rectangle.
* `lineType`: Type of the line. See #LineTypes
* `shift`: Number of fractional bits in the point coordinates.

```python
# pt1 = top left
# pt2 = bottom right
cv2.rectangle(blank_img,pt1=(384,0),pt2=(510,128),color=(0,255,0),thickness=5)
plt.imshow(blank_img)
```

![](opencv_resources/001-drawing1.png)



### Circles

```python
cv2.circle(img=blank_img, center=(100,100), radius=50, color=(255,0,0), thickness=5)
plt.imshow(blank_img)
```

![](opencv_resources/001-drawing2-circle.png)



### Filled in

```python
cv2.circle(img=blank_img, center=(400,400), radius=50, color=(255,0,0), thickness=-1)
plt.imshow(blank_img)
```

![](opencv_resources/001-drawing-filled-in.png)

### Line

```python
# Draw a diagonal blue line with thickness of 5 px
cv2.line(blank_img,pt1=(0,0),pt2=(511,511),color=(102, 255, 255),thickness=5)
plt.imshow(blank_img)
```

![](opencv_resources/001-drawing-line.png)

### Text

```python
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(blank_img,text='Hello',org=(10,500), fontFace=font,fontScale= 4,color=(255,255,255),thickness=2,lineType=cv2.LINE_AA)
plt.imshow(blank_img)
```

![](opencv_resources/001-drawing-text.png)

### Polygons

To draw a polygon, first you need coordinates of vertices. 

```python
blank_img = np.zeros(shape=(512,512,3),dtype=np.int32)
vertices = np.array([[100,300],[200,200],[400,300],[200,400]],np.int32)
pts = vertices.reshape((-1,1,2))
cv2.polylines(blank_img,[pts],isClosed=True,color=(255,0,0),thickness=5)
plt.imshow(blank_img)
```

![](opencv_resources/001-drawing-polygon.png)

### Fill Poly

```python
vertices = np.array([ [250,700], [425,400], [600,700] ], np.int32)
cv2.polylines(img, [vertices], isClosed=True, color=(0,0,255), thickness=10)
cv2.fillPoly(img, [vertices], color=(0,0,255))
plt.imshow(img)
```

![](opencv_resources/04-fill-poly.png)



# Image processing

## Color mapping

### Colorspaces

> Colorspaces là một mô hình toán học biểu diễn các màu sắc trong thực tế dưới dạng số học. Có rất nhiều Colorspaces: **BGR**, **BGRA**, **HSV**, **CMYK**,… Trong đó mỗi chữ cái đại diện cho 1 kênh màu (channel): **B**: Blue, **G**: Green, **R**: Red, **H**: Hue (màu sắc), S: Saturation (Độ bão hoà), **V**: Value, **A**: Alpha.

Màu sắc thường được mô tả trên hệ trục toạ độ, mỗi điểm trên hệ trục toạ độ đó tương ứng với một điểm.

Ex: Color space **RGB**

![](opencv_resources/0200-color-mapping1.webp)



### Converting to different colorspaces

Convert BGR (dùng trong opencv) sang RGB (hiển thị trong matplotlib)

```python 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
```

Convert to HSV

```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.imshow(img)
```

Convert to HLS

```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
plt.imshow(img)
```

