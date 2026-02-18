from sklearn.linear_model import LinearRegression
import numpy as np

# 모아놓은 데이터
# 보기 쉽게 3일씩 나누어서 썼음. 날마다 그날의 조회수를 적어준다.
views = np.array([144, 144, 146,
                  151, 155, 160,
                  164, 167, 171,
                  177, 178, 181,
                  183, 186, 187,
                  187, 187, 187])

diffs = np.diff(views)

# 학습 데이터 (3일 묶음)
X = []
y = []
for i in range(len(diffs) - 3):
    X.append(diffs[i:i+3])
    y.append(diffs[i+3])

X = np.array(X)
y = np.array(y)

model = LinearRegression()
model.fit(X, y)

# 예측용 입력(마지막 3일 증가폭)
last_window = diffs[-3:].reshape(1, -1)

# 다음날 증가폭 예측
pred_diff = model.predict(last_window)[0]

# 다음날 예측
pred_next = views[-1] + pred_diff
print("예측된 다음날 조회수: ", pred_next)