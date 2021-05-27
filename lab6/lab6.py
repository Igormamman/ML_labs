from sklearn.metrics._plot.confusion_matrix import plot_confusion_matrix
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, mean_absolute_percentage_error, confusion_matrix

@st.cache
def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv("/home/igor/Downloads/CarPrice_Assignment.xls",sep=',')
    return data


@st.cache
def preprocess_data(data_in):
    data_out = data_in.copy()
    cleanup_nums = {"doornumber":     {"four": 0, "two": 1},
                "cylindernumber": {"four": 4, "six": 6, "five": 5, "eight": 8,
                                  "two": 2, "twelve": 12, "three":3 },
                "fueltype" :       {"gas":0,"diesel":1},
                "aspiration":     {"std":0,"turbo":1},
                "enginelocation": {"front":0,"rear":1}}
    data_out = data_out.replace(cleanup_nums)
    data_out=pd.get_dummies(data_out, columns=["carbody"], prefix=["carbody"])
    data_out=pd.get_dummies(data_out, columns=["drivewheel"], prefix=["drive"])
    data_out["OHC_Code"] = np.where(data_out["enginetype"].str.contains("ohc"), 1, 0)
    data_out.drop(["CarName","enginetype","fuelsystem","symboling","car_ID"],axis=1,inplace=True)
    cols = data_out.columns
    sc1 = StandardScaler()
    data_out = sc1.fit_transform(data_out)
    data_out = pd.DataFrame(data_out,columns=cols)
    data_Y = data_out.loc[:, 'price']
    data_X = data_out.drop(["price"],axis=1,inplace=False)
    return  data_X, data_Y

main_status = st.text('')


data_load_state = st.text('Загрузка данных...')
data = load_data()
data_load_state.text('Данные загружены!')

data_load_state = st.text('Предварительная обработка данных...')
data_X,data_Y = preprocess_data(data)
data_load_state.text('Данные обработаны!')
#Количество записей
data_len = data_X.shape[0]


if st.checkbox('Показать корреляционную матрицу'):
    fig1, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    st.pyplot(fig1)

test_size = st.sidebar.slider("test_size", 0.1, 0.9, value = 0.3)
n_estimators = st.sidebar.slider("n_estimators", 1, 20, value=5)
n_neighbors = st.sidebar.slider("n_neighbors", 1, 20, value=5)
random_state = st.sidebar.slider("random_state", 1, 20, value=10)
max_depth = st.sidebar.slider("max_depth", 1, 10, value=4)

main_status.text('В процессе обучения...')

data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(
    data_X, data_Y,test_size=test_size, random_state=1)

# Модели
models_list = ['BagR', 'KNR', 'SVR', 'Tree', 'RF', 'GB']
clas_models = {'BagR': BaggingRegressor(n_estimators=n_estimators, random_state=random_state), 
               'KNR':KNeighborsRegressor(n_neighbors=n_neighbors),
               'SVR': LinearSVR(random_state=random_state),
               'Tree':DecisionTreeRegressor(max_depth=max_depth, random_state=random_state),
               'RF':RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=random_state),
               'GB':GradientBoostingRegressor(n_estimators = n_estimators, random_state=random_state),
               'AB':AdaBoostRegressor(n_estimators = n_estimators,random_state=random_state)}


main_status.text('Обучено!')

metrics = [mean_absolute_error, mean_squared_error, median_absolute_error, mean_absolute_percentage_error, confusion_matrix]
metr = [i.__name__ for i in metrics]
metrics_ms = st.sidebar.multiselect("Метрики", metr)
            
st.sidebar.header('Модели машинного обучения')
models_select = st.sidebar.multiselect('Выберите модели', models_list)

selMetrics = []
for i in metrics_ms:
    for j in metrics:
        if i == j.__name__:
            selMetrics.append(j)


st.header('Оценка')
for name in selMetrics:
    st.subheader(name.__name__)
    scorelist={}
    for model_name in models_select:
        model = clas_models[model_name]
        model.fit(data_X_train, data_y_train)
        # Предсказание значений
        Y_pred = model.predict(data_X_test)
        score =  name(Y_pred, data_y_test)
        scorelist[model_name]=score   
        st.text("{} - {}".format(model_name, score))
    if not scorelist:
        continue
    print(scorelist)
    scorelist = pd.DataFrame.from_dict(scorelist, orient = 'index')
    st.bar_chart(scorelist)