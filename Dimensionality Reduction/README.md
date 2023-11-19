# 차원축소

## Histogram of Oriented Gradients (HOG)
HOG는 이미지의 지역적 그래디언트 방향을 히스토그램으로 표현하여 이미지의 구조적 정보를 포착하는 특성 추출 방법   
HOG는 객체의 모양과 실루엣을 잘 잡아내는 데 특히 유용함   
![hog_image](https://github.com/Sumin0916/Fashionmnist_SVM/assets/95135403/281ad3fc-f183-4d45-8bf5-fd00f35cc3c2)




### apply_hog: hog를 적용함
```python
from skimage.feature import hog
def apply_hog(images):
    result = []
    for image in images:
        hog_features = hog(image, orientations=9, pixels_per_cell=(2, 2),
                        cells_per_block=(2, 2), block_norm='L2-Hys')
        result.append(hog_features)
    return result
```
* `orientations`: 한 픽셀이 어느 방향을 가리키는지를 나타내는 각도를 몇 개의 구간으로 나눌 것인지
  객체 인식 작업에서 좋은  성능을 보이는 값으로 알려져 있는 `9`로 설정함.
* `pixels_per_cell`: 각 셀의 크기.  (이미지의 세부 정보를 얼마나 캡처할 것인지)  
* `cells_per_block`: 블록 당 셀의 수. 
* `block_norm`: 블록 정규화 방법.

## PCA
```python
pca = PCA(n_components=250)
pca.fit(X_train_hog)
X_train_pca = pca.transform(X_train_hog)
y_test_pca = pca.transform(y_test_hog)

X_train_PCA1 = pd.DataFrame(X_train_pca)
X_test_PCA1 = pd.DataFrame(y_test_pca)
```
Hog 적용 후 PCA 적용    
이전 값 `n_components=400` 사용시 과적합 문제 발생하여 `n_components=250`으로 실행하였음    
(같은 문제로 svm의 C값도 `5`로 설정하였음)

## calculate_mAP : 예측과 label 비교하여 최종 mAP출력

제공된 파일에서 로직은 그대로 가져다 씀
* `label`: 훈련 데이터와 테스트 데이터의 label중 무엇을 쓸것인지
* 최종 mAP 계산결과만 출력함

```python
from sklearn.metrics import auc
from collections import Counter
def calculate_mAP(preds,label):
    ## mAP calculation
    AP = []
    num_class = 10
    predict_label_count_dict = Counter(preds)
    predict_label_count_dict = dict(sorted(predict_label_count_dict.items()))

    # For each class
    for c, freq in predict_label_count_dict.items() :
        TP = 0
        FN = 0

        temp_precision = []
        temp_recall = []

        for i in range(len(preds)):
            # Calculate TP and FN
            if label[i] == c and preds[i] == c :
                TP += 1
            elif label[i] != c and preds[i] == c :
                FN += 1

            # Calculate precision and recall
            if TP+FN != 0:
                temp_precision.append(TP/(TP+FN))
                temp_recall.append(TP/freq)

        # Save the AP value of each class to AP array
        AP.append(auc(temp_recall, temp_precision))

    # Calculate mAP
    mAP = sum(AP) / num_class

    return mAP
```

## Result
Hog를 쓴다면 과적합 문제를 해결해야할듯 함.    
![스크린샷 2023-11-19 220520](https://github.com/Sumin0916/Fashionmnist_SVM/assets/95135403/7c926250-dd47-4001-b515-2f66bb5561c5)    
![스크린샷 2023-11-19 220549](https://github.com/Sumin0916/Fashionmnist_SVM/assets/95135403/aee7daa8-23b5-46cb-b8e1-8f5c42e9836a)
