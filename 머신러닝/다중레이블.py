import numpy as np
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, accuracy_score

# 1. 다중 레이블 데이터 생성
# n_samples: 샘플 수
# n_features: 특성 수
# n_classes: 가능한 총 클래스(레이블) 수
# n_labels: 각 샘플이 가질 수 있는 평균 레이블 수
X, y = make_multilabel_classification(n_samples=100, n_features=20, n_classes=5,
                                      n_labels=2, random_state=42)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("\n첫 5개 샘플의 레이블 (이진 행렬):")
print(y[:5])

# 2. 훈련 및 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. MultiOutputClassifier와 SVM 사용
# MultiOutputClassifier는 각 출력(레이블)에 대해 독립적인 분류기를 훈련합니다.
# 내부적으로 One-vs-Rest 전략과 유사하게 작동합니다.
base_svm = SVC(kernel='linear', probability=True, random_state=42) # 확률 예측을 위해 probability=True 설정
multi_label_svm = MultiOutputClassifier(base_svm, n_jobs=-1) # n_jobs=-1로 병렬 처리 (가능한 경우)

# 4. 모델 훈련
print("\n모델 훈련 시작...")
multi_label_svm.fit(X_train, y_train)
print("모델 훈련 완료.")

# 5. 예측
y_pred = multi_label_svm.predict(X_test)
y_pred_proba = multi_label_svm.predict_proba(X_test) # 확률 예측

print("\n실제 레이블 (첫 5개):")
print(y_test[:5])
print("\n예측된 레이블 (첫 5개):")
print(y_pred[:5])

# 6. 모델 평가
# 다중 레이블 분류에서는 여러 평가 지표를 사용할 수 있습니다.

# 햄밍 손실 (Hamming Loss):
# 잘못 예측된 레이블의 비율. 0에 가까울수록 좋습니다.
h_loss = hamming_loss(y_test, y_pred)
print(f"\nHamming Loss: {h_loss:.4f}")

# 정확도 (Exact Match Ratio / Jaccard Similarity Score):
# 모든 레이블을 정확히 예측한 샘플의 비율 (엄격한 지표).
# Scikit-learn 0.23부터 jaccard_score가 사라지고 jaccard_similarity_score로 대체되었으나,
# 1.0.0 이후로 jaccard_score로 다시 통합되었습니다.
from sklearn.metrics import jaccard_score

# average='samples'는 각 샘플에 대해 정확히 일치하는 레이블의 비율을 계산한 후 평균을 냅니다.
jaccard_similarity = jaccard_score(y_test, y_pred, average='samples')
print(f"Jaccard Similarity Score (Exact Match Ratio): {jaccard_similarity:.4f}")

# F1-스코어 (Micro/Macro F1-score):
# 다중 레이블에서 자주 사용되는 평가 지표입니다.
from sklearn.metrics import f1_score

# micro: 전체 TP, FP, FN을 합산하여 계산 (레이블 불균형에 덜 민감)
f1_micro = f1_score(y_test, y_pred, average='micro')
print(f"Micro F1-score: {f1_micro:.4f}")

# macro: 각 레이블에 대한 F1-score를 계산한 후 평균 (레이블 불균형에 민감)
f1_macro = f1_score(y_test, y_pred, average='macro')
print(f"Macro F1-score: {f1_macro:.4f}")

# 각 레이블에 대한 예측 확률 확인 (예시)
# print("\n첫 번째 테스트 샘플에 대한 각 레이블의 예측 확률:")
# for i, prob in enumerate(y_pred_proba):
#    print(f"Label {i}: {prob[0]:.4f} (class 0, no), {prob[1]:.4f} (class 1, yes)")