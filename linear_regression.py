import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import r2_score

# Загрузка данных
file_path = 'data' # Укажите путь к вашему файлу данных
target_column = 'target' # Укажите целевой столбец
data = pd.read_csv(file_path)
X = data.drop(target_column, axis=1)
y = data[target_column]

# def load_data(file_path, target_column):
#     file_extension = file_path.split('.')[-1]
#     if file_extension == 'csv':
#         data = pd.read_csv(file_path)
#     elif file_extension == 'pkl':
#         with open(file_path, 'rb') as f:
#             data = pickle.load(f)
#     elif file_extension in ['xls', 'xlsx']:
#         data = pd.read_excel(file_path)
#     else:
#         raise ValueError("Неподдерживаемый формат файла. Поддерживаемые форматы: CSV, PKL, XLS, XLSX")
#     X = data.drop(target_column, axis=1)
#     y = data[target_column]
#     return X, y


# Предобработка данных
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])
categorical_features = X.select_dtypes(include=['object', 'bool', 'category']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Заполнение пропусков наиболее частыми значениями
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])  # Кодирование категориальных переменных

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# обнаружение и удаление выбросов
outlier_detector = IsolationForest(random_state=42, contamination=0.01)  # 1% данных считаются выбросами
outliers = outlier_detector.fit_predict(preprocessor.fit_transform(X))
X_filtered = X[outliers != -1]
y_filtered = y[outliers != -1]


# Разбиение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# Определение модели и параметров для GridSearch
model = {
    'LinearRegression': {
        'model': Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', LinearRegression())]),
        'params': {
            'regressor__fit_intercept': [True, False],
        }
    }
}

best_models = {}
results = []

# Подбор гиперпараметров и обучение модели
for name, spec in models.items():
    grid_search = GridSearchCV(spec['model'], spec['params'], cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_models[name] = best_model
    joblib.dump(best_model, f'{name}_best_model.joblib')  # Сохранение лучшей модели

    # Вычисление метрик для лучшей модели
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - RMSE: {rmse}, MAE: {mae}, R2: {r2}")

    # # Отчет о кросс-валидации
    # scores = cross_val_score(best_model, X_train, y_train, cv=5)
    # print(f"{name} - Средняя точность кросс-валидации: {scores.mean()}")

