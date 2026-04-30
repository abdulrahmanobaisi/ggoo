from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# تحميل الداتا
data = load_iris()
X = data.data
y = data.target

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# إنشاء الـ Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # Step 1: Normalize1
    ('model', LogisticRegression())     # Step 2: Model
])

# تدريب الموديل
pipeline.fit(X_train, y_train)

# التقييم
accuracy = pipeline.score(X_test, y_test)
print("Accuracy:", accuracy)