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

![](opencv_resources/colorspace1.png)

Convert to HSV

```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.imshow(img)
```

![](opencv_resources/colorspace2.png)

Convert to HLS

```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
plt.imshow(img)
```

![](opencv_resources/colorspace3.png)



## Blending and Pasting images

For some computer vision systems, we'll want to be able to post our own image on top of an already existing image or video. We may also want to blend images, maybe we want to have a "highlight" effect instead of just a solid box or empty rectangle.

Let's explore what is commonly known as ***\*Arithmetic Image Operations\**** with OpenCV. These are referred to as Arithmetic Operations because OpenCV is simply performing some common math with the pixels for the final effect. We'll see a bit of this in our code.

### Blending basic

```python
# Two images
img1 = cv2.imread('../DATA/dog_backpack.png')
img2 = cv2.imread('../DATA/watermark_no_copy.png')

# Convert to RGB space
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Resizing image
img1 =cv2.resize(img1,(1200,1200))
img2 =cv2.resize(img2,(1200,1200))
```

**Blending the image**

We will blend the values together with the formula:

$$  img_1 * \alpha  + img_2 * \beta  + \gamma $$

```python
blended = cv2.addWeighted(src1=img1,alpha=0.7,src2=img2,beta=0.3,gamma=0)
plt.imshow(blended)
```

![](opencv_resources/blending-image1.png)

### Overlaying images of different sizes

We can use this quick trick to quickly overlap different sized images, by simply reassigning the larger image's values to match the smaller image.

```python
# Load two images
img1 = cv2.imread('../DATA/dog_backpack.png') #shape: (1401, 934, 3)
img2 = cv2.imread('../DATA/watermark_no_copy.png') #shape: (1280, 1277, 3)
img2 =cv2.resize(img2,(600,600))

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

large_img = img1
small_img = img2

x_offset=0
y_offset=0

large_img[y_offset:y_offset+small_img.shape[0], x_offset:x_offset+small_img.shape[1]] = small_img
plt.imshow(large_img)
```

![](opencv_resources/blending2.png)

### Blending images of different sizes

```python
# Load two images
img1 = cv2.imread('../DATA/dog_backpack.png') #shape: (1401, 934, 3)
img2 = cv2.imread('../DATA/watermark_no_copy.png') #shape: (1280, 1277, 3)
img2 =cv2.resize(img2,(600,600))

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
```

#### Create a Region of Interest (ROI)

```python
x_offset=934-600
y_offset=1401-600

# Creating an ROI of the same size of the foreground image (smaller image that will go on top)
rows,cols,channels = img2.shape
# roi = img1[0:rows, 0:cols ] # TOP LEFT CORNER
roi = img1[y_offset:1401,x_offset:943] # BOTTOM RIGHT CORNER
plt.imshow(roi)
```

![](opencv_resources/blending3.png)

#### Creating a Mask

```python
# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
img2gray.shape # (600, 600)
plt.imshow(img2gray,cmap='gray')
```

![](opencv_resources/bleding4.png)

```python
mask_inv = cv2.bitwise_not(img2gray)
mask_inv.shape # (600, 600)
plt.imshow(mask_inv,cmap='gray')
```

![](opencv_resources/blending5.png)

#### Convert mask to have 3 channels

```python
white_background = np.full(img2.shape, 255, dtype=np.uint8)
bk = cv2.bitwise_or(white_background, white_background, mask=mask_inv)
bk.shape # (600, 600, 3)
plt.imshow(bk)
```

![](opencv_resources/blending6.png)

#### Grab original forceground image and place on top of Mask

```python
fg = cv2.bitwise_or(img2, img2, mask=mask_inv)
fg.shape # (600, 600, 3)
plt.imshow(fg)
```

![](opencv_resources/blending7.png)

#### Get ROI and blend in the mask with the ROI

```python
final_roi = cv2.bitwise_or(roi,fg)
plt.imshow(final_roi)
```

![](opencv_resources/blending8.png)

#### Add in the rest of the image

```python
large_img = img1
small_img = final_roi
large_img[y_offset:y_offset+small_img.shape[0], x_offset:x_offset+small_img.shape[1]] = small_img
plt.imshow(large_img)
```

![](opencv_resources/blending9.png)

## Image thresholding

Trong opencv, ngưỡng là một số nằm trong đoạn từ 0 đến 255. Giá trị ngưỡng sẽ chia tách giá trị độ xám của ảnh thành 2 miền riêng biệt. Miền thứ nhất là tập hợp các điểm ảnh có giá trị nhỏ hơn giá trị ngưỡng. Miền thứ hai là tập hợp các các điểm ảnh có giá trị lớn hơn hoặc bằng giá trị ngưỡng.

Nếu pixel có giá trị lớn hơn giá trị ngưỡng thì nó được gán 1 giá trị (thường là 1), ngược lại nhỏ hơn giá trị ngưỡng thì nó được gán 1 giá trị khác (thường là 0).

`cv.threshold(src, thresh, maxval, type[, dst]) ->retval, dst`

- **src**: input array (multiple-channel, 8-bit or 32-bit floating point)
- **dst**: output array of the same size and type and the same number of channels as src.
- **thresh**:  threshold value.
- **maxval**:  maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
- **type**:    thresholding type (see ThresholdTypes).

```python
img = cv2.imread('../DATA/rainbow.jpg')
plt.imshow(img)
```

![](opencv_resources/thresholing1.png)


```python
plt.imshow(img,cmap='gray')
```

![](opencv_resources/thresholing2.png)

### Different threshold types

#### Binary

- Nếu giá trị pixel lớn hơn ngưỡng thì gán bằng maxval
- Ngược lại bằng gán bằng 0

```python
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret # 127.0
plt.imshow(thresh1,cmap='gray')
```

![](opencv_resources/thresholing3.png)

#### Binary Inverse

- Nếu giá trị pixel lớn hơn ngưỡng thì gán bằng 0
- Ngược lại bằng gán bằng maxval

```python
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
plt.imshow(thresh2,cmap='gray')
```

![](opencv_resources/thresholing4.png)

#### Threshold Truncation

- Nếu giá trị pixel lớn hơn ngưỡng thì gán giá trị bằng ngưỡng
- Ngược lại giữ nguyên giá trị

```python
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
plt.imshow(thresh3,cmap='gray')
```

![](opencv_resources/thresholing5.png)

#### Threshold to Zero

- Nếu giá trị pixel lớn hơn ngưỡng thì giữ nguyên giá trị
- Ngược lại gán bằng 0

```python
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
plt.imshow(thresh4,cmap='gray')
```

![](opencv_resources/thresholing6.png)

#### Threshold to Zero (Inverse)

- Nếu giá trị pixel lớn hơn ngưỡng thì gán giá trị bằng 0
- Ngược lại giữ nguyên

```python
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
plt.imshow(thresh5,cmap='gray')
```

![](opencv_resources/thresholing7.png)

### Real World application

**Sodoku image**

```python
img = cv2.imread("../DATA/crossword.jpg",0)

def show_pic(img):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
    
show_pic(img)
```

![](opencv_resources/thresholing8.png)

#### Simple Binary

```python
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
show_pic(th1)
```

![](opencv_resources/thresholing9.png)

#### Adaptive threshold

Thuật toán simple thresholding hoạt động khá tốt. Tuy nhiên, nó có 1 nhược điểm là giá trị ngưỡng bị/được gán toàn cục. Thực tế khi chụp, hình ảnh chúng ta nhận được thường bị ảnh hưởng của nhiễu, ví dụ như là bị phơi sáng, bị đèn flask, …

Một trong những cách được sử dụng để giải quyết vấn đề trên là chia nhỏ bức ảnh thành những vùng nhỏ (region), và đặt giá trị ngưỡng trên những vùng nhỏ đó -> adaptive thresholding ra đời. Opencv cung cấp cho chúng ta hai cách xác định những vùng nhỏ

https://stackoverflow.com/questions/28763419/adaptive-threshold-parameters-confusion



​    @param `src` Source 8-bit single-channel image.

​    @param `dst` Destination image of the same size and the same type as src.

​    @param `maxValue` Non-zero value assigned to the pixels for which the condition is satisfied

​    @param `adaptiveMethod` Adaptive thresholding algorithm to use, see `#AdaptiveThresholdTypes`.

​     The `#BORDER_REPLICATE` | `#BORDER_ISOLATED` is used to process boundaries.

​    @param `thresholdType` Thresholding type that must be either `#THRESH_BINARY` or `#THRESH_BINARY_INV`,

​     see `#ThresholdTypes`.

​    @param `blockSize` Size of a pixel neighborhood that is used to calculate a threshold value for the

​    pixel: 3, 5, 7, and so on.

​    @param `C` Constant subtracted from the mean or weighted mean (see the details below). Normally, it

​    is positive but may be zero or negative as well.

```python
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8) # Play around with these last 2 numbers
show_pic(th2)
```

![](opencv_resources/thresholing10.png)

```python
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,8)
```

![](/Users/ngohongthai/Documents/projects/funix/docs/contents/c4-deep-learning/opencv_resources/thresholing11.png)

```python
blended = cv2.addWeighted(src1=th1,alpha=0.7,src2=th2,beta=0.3,gamma=0)
show_pic(blended)
```

![](opencv_resources/thresholing12.png)

## Blurring and Smoothing

There are a lot of different effects and filters we can apply to images. We're just going to go through a few of them here. Most of them involve some sort of math based function being applied to all the pixels values.

\-----

**Info Link on Blurring Math:**

http://people.csail.mit.edu/sparis/bf_course/

-----

### **Convenience Functions**

Quick function for loading the puppy image.
