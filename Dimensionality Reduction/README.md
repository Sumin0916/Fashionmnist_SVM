# 차원축소

## Histogram of Oriented Gradients (HOG)
HOG는 이미지의 지역적 그래디언트 방향을 히스토그램으로 표현하여 이미지의 구조적 정보를 포착하는 특성 추출 방법   
HOG는 객체의 모양과 실루엣을 잘 잡아내는 데 특히 유용함   
![hog_image](https://github.com/Sumin0916/Fashionmnist_SVM/assets/95135403/281ad3fc-f183-4d45-8bf5-fd00f35cc3c2)




### Parameters
```apply_hog```
* `orientations`: 한 픽셀이 어느 방향을 가리키는지를 나타내는 각도를 몇 개의 구간으로 나눌 것인지
  객체 인식 작업에서 좋은  성능을 보이는 값으로 알려져 있는 `9`로 설정함.
* `pixels_per_cell`: 각 셀의 크기.  (이미지의 세부 정보를 얼마나 캡처할 것인지)  
* `cells_per_block`: 블록 당 셀의 수. 
* `block_norm`: 블록 정규화 방법.

## PCA
이전 값 `n_components=400` 사용시 과적합 문제 발생하여 `n_components=250`으로 실행하였음    
(같은 문제로 svm의 C값도 `5`로 설정하였음)

## Result





