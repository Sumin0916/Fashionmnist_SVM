# 하이퍼파리미터 튜닝
## Optuna
## objective - Map 성능으로 PCA의 n_componetns, SVM의 gamma, C값을 튜닝
`kernel`은 시간이 오래걸리는 관계로  지정해주었음 
* `rbf`: `optuna_rbf.ippynb`에서 확인가능
* `poly`: 실행한지 이틀되었으나 결과 아직 안나옴.. `trials` 줄여야할듯 현재는 `50`번
``

```
def objective(trial):
    # suggest methods are used to set the range of hyperparameters
    n_components = trial.suggest_int('n_components', 181, 600)
    gamma = trial.suggest_loguniform('gamma', 1e-3, 1e+2)
    C = trial.suggest_loguniform('C', 1e+0, 1e+4)

    pca = PCA(n_components=n_components)
    svm = SVC(gamma=gamma, C=C, kernel="poly", probability=True)

    # PCA fitting and transformation should be done inside the objective function
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)

    # Use transformed data to train the classifier
    svm.fit(X_train_pca, y_train)

    # Predict the validation data
    preds = svm.predict(X_val_pca)

    ## mAP calculation
    AP = []
    num_class = 10

    # Count the number of each class in preds
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
            if y_val[i] == c and preds[i] == c :
                TP += 1
            elif y_val[i] != c and preds[i] == c :
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
* 튜닝 결과    
![스크린샷 2023-11-19 231651](https://github.com/Sumin0916/Fashionmnist_SVM/assets/95135403/07dbc0f2-6ba4-40d2-b5d4-013c65f254b9)


* 튜닝 기록    
![newplot](https://github.com/Sumin0916/Fashionmnist_SVM/assets/95135403/2a7dd620-de93-4963-ade4-4ea39900933a)    
* 파라미터 사이의 관계    
![newplot1](https://github.com/Sumin0916/Fashionmnist_SVM/assets/95135403/a5a9beee-2bb5-4cbc-99cc-5b1dd52a981c)    
* 파라미터 중요도    
![newplot2](https://github.com/Sumin0916/Fashionmnist_SVM/assets/95135403/5be8a0e3-ab99-434f-b685-3b9bbc8080ab)

