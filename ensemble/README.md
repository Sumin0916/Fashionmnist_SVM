# Ensemble에 쓸만한 모델 모음
# Table of Contents
1. [LightGBM](#lightgbm)
2. [CatBoost](#catboost)
## LightGBM 

 LightGBM은 Gradient Boosting 프레임워크를 기반으로 하는 머신 러닝 알고리즘

- **속도와 메모리 효율성**: LightGBM은 트리 분할 방법으로 'Leaf-wise' 방식을 사용하여 빠른 학습 속도,적은 메모리 사용량
- **높은 성능**: LightGBM은 높은 정확도를 달성할 수 있으며, 대용량 데이터 처리에도 잘 동작함
- **범주형 변수 처리**: LightGBM은 범주형 변수를 자동으로 처리할 수 있는 기능을 제공

```python
lgbm_model = LGBMClassifier(
    n_estimators=839,
    num_leaves=4,
    min_child_samples=6,
    learning_rate=0.17305095027775025,
    max_bin=1024,  # log_max_bin을 10으로 설정했으므로, max_bin은 2^10인 1024가 됩니다.
    colsample_bytree=0.8717502271722275,
    reg_alpha=0.036114468962103394,
    reg_lambda=0.23607505416113697,
)
```

- `n_estimators`: 부스팅 스테이지의 수. 스테이지가 많을수록 모델은 복잡해지고, 과적합의 위험이 증가.
- `num_leaves`: 트리의 최대 잎의 수. 이 값이 클수록 모델은 복잡해지고, 과적합의 위험이 증가
- `min_child_samples`: 리프 노드가 되기 위한 최소 데이터 수. 이 값이 클수록 과적합을 방지하는 데 도움이 됨
- `learning_rate`: 학습률. 작을수록 모델 학습은 느려지나 더 좋은 성능 달성할 가능성
- `max_bin=1024`: 특성 값을 분할하는 데 사용하는 최대 bin의 수. 이 값이 클수록 모델은 더 정확해질 수 있지만, 과적합의 위험이 증가하고 학습 속도가 느려질 수 있음
- `colsample_bytree`: 트리를 학습시킬 때 특성의 일부를 샘플링하는 비율
- `reg_alpha`: L1 정규화 항의 가중치. 이 값이 클수록 모델은 더 간단해지고, 과적합을 방지하는 데 도움이 될 수 있음
- `reg_lambda`: L2 정규화 항의 가중치. 이 값이 클수록 모델은 더 간단해지고, 과적합을 방지하는 데 도움이 될 수 있음
- [파라미터 값 참고](https://www.kaggle.com/code/gauravduttakiit/fashion-mnist-classifier-flaml-micro-f1)
### XGBoost vs lightGBM
* LightGBM은 학습 속도와 메모리 효율성 면에서 우수한 성능을 보여줌
* 그러나 LightGBM은 더 깊은 트리를 사용하기 때문에 과적합의 위험
* 복잡한 데이터셋에 적합
* 
## CatBoost

Gradient Boosting 알고리즘에 기반한 이 방법. 특히 범주형 데이터(Categorical Data)를 다루는 데 뛰어난 성능을 보임   
fahsionMNIST는 범주형 데이터가 아니라는 점 고려해야함 (덜 추천)

- **범주형 데이터 처리**: CatBoost는 범주형 변수를 자동으로 처리하므로, 별도의 데이터 전처리 없이 범주형 변수를 사용가능
- **우수한 성능**: Gradient Boosting 알고리즘에 기반, CatBoost는 다른 머신러닝 알고리즘에 비해 뛰어난 성능
- **체크포인트 저장**: 학습 도중에 체크포인트를 저장, 학습이 중단되더라도 이어서 학습 가능

```python
catboost = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6)
```

- `iterations`: 학습을 반복할 횟수. 이 값이 크면 더 많이 학습하지만, 과적합의 위험.
- `learning_rate`: 학습률. 이 값이 너무 크면 학습이 불안정할 수 있고, 너무 작으면 학습이 느려짐.
- `depth`: 트리의 깊이.이 값이 크면 더 복잡한 모델을 만들 수 있지만, 과적합의 위험.


