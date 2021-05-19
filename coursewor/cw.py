from IPython.display import Image
import numpy as np
from numpy.lib import load
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import statsmodels.regression.linear_model as sm
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score , recall_score , plot_confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import  StandardScaler,MinMaxScaler
sns.set(style="ticks")


class MetricLogger:
    
    def __init__(self):
        self.df = pd.DataFrame(
            {'metric': pd.Series([], dtype='str'),
            'alg': pd.Series([], dtype='str'),
            'value': pd.Series([], dtype='float')})

    def add(self, metric, alg, value):
        """
        Добавление значения
        """
        # Удаление значения если оно уже было ранее добавлено
        self.df.drop(self.df[(self.df['metric']==metric)&(self.df['alg']==alg)].index, inplace = True)
        # Добавление нового значения
        temp = [{'metric':metric, 'alg':alg, 'value':value}]
        self.df = self.df.append(temp, ignore_index=True)

    def get_data_for_metric(self, metric, ascending=True):
        """
        Формирование данных с фильтром по метрикеопроб2
        """
        temp_data = self.df[self.df['metric']==metric]
        temp_data_2 = temp_data.sort_values(by='value', ascending=ascending)
        return temp_data_2['alg'].values, temp_data_2['value'].values
    
    def plot(self, str_header, metric, ascending=True, figsize=(5, 5)):
        """
        Вывод графика
        """
        array_labels, array_metric = self.get_data_for_metric(metric, ascending)
        fig, ax1 = plt.subplots(figsize=figsize)
        pos = np.arange(len(array_metric))
        rects = ax1.barh(pos, array_metric,
                         align='center',
                         height=0.5, 
                         tick_label=array_labels)
        ax1.set_title(str_header)
        for a,b in zip(pos, array_metric):
            plt.text(0.5, a-0.05, str(round(b,3)), color='white') 
        st.pyplot(fig)
        return(fig)    

@st.cache
def load_data():
    '''
    Загрузка данных
    '''
    #Считываем датасет из csv файла.
    data = pd.read_csv("/home/igor/Downloads/Stars.csv", sep=',')
    # Удалим дубликаты записей, если они присутствуют.
    data = data.drop_duplicates()
    return data

@st.cache
def preprocess_data(data_in):
    data_out = data_in.copy()
    cleanup_nums = {"Color": {"Red": 0, "Blue White": 1,
                              "White": 2,"Yellowish White": 4,
                               "Blue white": 1,
       "Pale yellow orange": 5, "Blue": 6, "Blue-white": 1, "Whitish": 2,
       "yellow-white":4, "Orange": 5, "White-Yellow": 4, "white": 2, "yellowish": 8,
       "Yellowish": 8, "Orange-Red": 0, "Blue-White": 1 },
    }
    data_out = data_out.replace(cleanup_nums)
    data_out=pd.get_dummies(data_out, columns=["Spectral_Class"], prefix=["Class"])
    data_out = data_out.replace(cleanup_nums)
    data_out = data_out.drop(['Class_O'],axis=1)
    data_out= data_out.drop(['Class_B'],axis=1)
    # Числовые колонки для масштабирования
    data_scaled = data_out.copy()
    scale_cols = ['L', 'R', 'A_M']
    sc1 = StandardScaler()
    sc1_data = sc1.fit_transform(data_out[scale_cols])
    # Добавим масштабированные данные в набор данных
    for i in range(len(scale_cols)):
        col = scale_cols[i]
        new_col_name = col + '_scaled'
        data_scaled[new_col_name] = sc1_data[:,i]
   
       
    data_Y = data_out.loc[:, 'Type']
    data_X = data_out.drop(["Type"],axis=1,inplace=False)
    return  data_X, data_Y

@st.cache
def scale_data(data_in):
    data_out = data_in.copy()
    # Числовые колонки для масштабирования
    data_scaled = data_out.copy()
    scale_cols = ['L', 'R', 'A_M']
    sc1 = StandardScaler()
    sc1_data = sc1.fit_transform(data_out[scale_cols])
    # Добавим масштабированные данные в набор данных
    for i in range(len(scale_cols)):
        col = scale_cols[i]
        new_col_name = col + '_scaled'
        data_scaled[new_col_name] = sc1_data[:,i]
    for col in scale_cols:
        col_scaled = col + '_scaled'
        fig, ax = plt.subplots(1, 2, figsize=(8,3))
        ax[0].hist(data_scaled[col], 50)
        ax[1].hist(data_scaled[col_scaled], 50)
        ax[0].title.set_text(col)
        ax[1].title.set_text(col_scaled)
    return  fig

# Function to calculate VIF
@st.cache
def calculate_vif(data):
    vif_df = pd.DataFrame(columns = ['Var', 'Vif'])
    x_var_names = data.columns
    for i in range(0, x_var_names.shape[0]):
        y = data[x_var_names[i]]
        x = data[x_var_names.drop([x_var_names[i]])]
        r_squared = sm.OLS(y,x).fit().rsquared
        vif = round(1/(1-r_squared),2)
        vif_df.loc[i] = [x_var_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis = 0, ascending=False, inplace=False)

def clas_train_model(model_name, model, clasMetricLogger):
    model.fit(data_X_train, data_y_train)
    # Предсказание значений1
    Y_pred = model.predict(data_X_test)
    
    accuracy = accuracy_score(data_y_test, Y_pred)
    
    clasMetricLogger.add('accuracy', model_name, accuracy)
    fig, ax = plt.subplots(ncols=1, figsize=(10,5))  
    plot_confusion_matrix(model, data_X_test, data_y_test.values,ax=ax,
                      display_labels=data_Y.unique(), 
                      cmap=plt.cm.Blues)
    fig.suptitle(model_name)
    print('{} \t Accuracy={}'.format( model_name,round(accuracy, 3)))
    st.pyplot(fig)
    

if st.checkbox('Показать анализ данных'):
    """
        Temperature -- K

        L -- L/Lo

        R -- R/Ro

        AM -- Mv

        Color -- General Color of Spectrum

        Spectral_Class -- O,B,A,F,G,K,M / SMASS - https://en.wikipedia.org/wiki/Asteroid_spectral_types

        Type -- Red Dwarf, Brown Dwarf, White Dwarf, Main Sequence , Super Giants, Hyper Giants

        MATH:

        Lo = 3.828 x 10^26 Watts

        (Avg Luminosity of Sun)

        Ro = 6.9551 x 10^8 m

        (Avg Radius of Sun)"""
    
    data=load_data()
    st.write("Количество строк :{}, количество столбцов : {}".format(data.shape[0],data.shape[1]))
    st.text("Количество пустых полей:") 
    st.dataframe(data.isnull().sum())
    st.dataframe(data.head())
    cleanup_nums = {"Color":     {"Red": 0, "Blue White": 1,
                              "White": 2,"Yellowish White": 4,
                               "Blue white": 1,
       "Pale yellow orange": 5, "Blue": 6, "Blue-white": 1, "Whitish": 2,
       "yellow-white":4, "Orange": 5, "White-Yellow": 4, "white": 2, "yellowish": 8,
       "Yellowish": 8, "Orange-Red": 0, "Blue-White": 1 },
    }
    data = data.replace(cleanup_nums)
    data=pd.get_dummies(data, columns=["Spectral_Class"], prefix=["Class"])
   
    st.write(data.dtypes)
  
    X=data.drop(['Type'],axis=1)
    
    st.write(calculate_vif(X))
   
    """Так как коэффиициент мультиколлинеарности для параметра для классов ClassO и СlassB больше либо примерно равно 5,
     то удалим их, и пересчитаем коэффициенты мультиколлинеарности"""

    data = data.drop(['Class_O'],axis=1)
    data= data.drop(['Class_B'],axis=1)
    X=data.drop(['Type'],axis=1)

    st.write(calculate_vif(X))

    """Вывод"""


  
    """Построим графики для понимания структуры данных"""

    """Для начала оценим дизбаланс классов относительно целевого признака"""

    # Оценим дисбаланс классов для Occupancy
    fig, ax = plt.subplots(figsize=(2,2)) 
    plt.hist(data['Type'])
    st.pyplot(fig)

    """Из графика очевидно, что дизбаланс классов отсутствует"""

    """Построим парные диаграммы для рассматриваемого набора данных"""
    fig, ax = plt.subplots(figsize=(15,15)) 
    #fig = sns.pairplot(data)
    #st.pyplot(fig)

    """Построим скрипичные диаграммы для числовых параметров"""
    for col in ['L', 'R', 'Color', 'A_M']:
        sns.violinplot(x=data[col])
        plt.show()

    st.title("""Выбор признаков, подходящих для построения моделей.Масштабирование данных. Формирование вспомогательных признаков, улучшающих качество моделей.""")

    """Выполним масштабирование данных, и проверим, что масштабирование не повлияло на распределение данных"""

    # Добавим масштабированные данные в набор данных
    data_scaled = data.copy()
    scale_cols = ['L', 'R', 'A_M']
    sc1 = StandardScaler()
    sc1_data = sc1.fit_transform(data[scale_cols])
    for i in range(len(scale_cols)):
        col = scale_cols[i]
        new_col_name = col + '_scaled'
        data_scaled[new_col_name] = sc1_data[:,i]
   
    # Проверим, что масштабирование не повлияло на распределение данных
    for col in scale_cols:
        col_scaled = col + '_scaled'
        fig, ax = plt.subplots(1, 2, figsize=(8,3))
        ax[0].hist(data_scaled[col], 50)
        ax[1].hist(data_scaled[col_scaled], 50)
        ax[0].title.set_text(col)
        ax[1].title.set_text(col_scaled)
        st.pyplot(fig)

    st.title("Проведение корреляционного анализа данных. Формирование промежуточных выводов о возможности построения моделей машинного обучения.")

    data_scaled = data_scaled.drop(scale_cols,axis=1)
    """Набор данных полученный после масштабирования данных"""
    st.dataframe(data_scaled.head())
    """Построим корелляционную матрицу для масштабированных данных, чтобы убедиться что зависимости не были нарушены"""
    fig1, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    st.pyplot(fig1)
    
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(data_scaled.corr(), annot=True, fmt='.2f')
    st.pyplot(fig)

    """На основе корреляционной матрицы можно сделать следующие выводы:

    -  Корреляционные матрицы для исходных и масштабированных данных совпадают(за исключением поряка строк и столбцов).
    -  Целевой признак классификации "Type" наиболее сильно коррелирует с численными параметрами такими как Радиус(R)(0.66), Светимостью(L)(0.68) и A_M (-0.96). Эти признаки обязательно следует оставить в моделирам классификации.
    -  Сильнокореллированных признаков нет.
    -  Большие по модулю значения коэффициентов корреляции свидетельствуют о значимой корреляции между исходными признаками и целевым признаком. На основании корреляционной матрицы можно сделать вывод о том, что данные позволяют построить модель машинного обучения."""


    st.title("Выбор метрик для последующей оценки качества моделей.")
    """В качестве метрик для решения задачи классификации будем использовать следующие метрики:

    - confusion matrix т.к. позволяет визуально наблюдать результаты классификации
    - accuracy для численного отображения результаттов классификации"""

    st.title("Выбор наиболее подходящих моделей для решения задачи классификации или регрессии.")

    """Для задачи классификации будем использовать следующие модели:
    
    -   Логистическая регрессия
    -   Метод ближайших соседей
    -   Метод опорных векторов
    -   Решающее дерево
    -   Случайный лес
    -   Градиентный бустинг """

    st.title("Формирование обучающей и тестовой выборок на основе исходного набора данных.")

    data_Y = data.loc[:, 'Type']
    data_X = data.drop(["Type"],axis=1,inplace=False)
    data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(
    data_X, data_Y,test_size=0.6, random_state=360)

    st.write("""Количество строк в тестовой выборке: {},
    Количество строк в тренировочной выборке:{}""".format(data_X_test.shape[0],data_y_test.shape[0]))

    st.title("Построение базового решения (baseline) для выбранных моделей без подбора гиперпараметров. Производится обучение моделей на основе обучающей выборки и оценка качества моделей на основе тестовой выборки.")

    clas_models = {'LogR': LogisticRegression(), 
               'KNN_5':KNeighborsClassifier(n_neighbors=5),
               'SVC':SVC(probability=True),
               'Tree':DecisionTreeClassifier(),
               'RF':RandomForestClassifier(),
               'GB':GradientBoostingClassifier()}
    
    clasMetricLogger = MetricLogger()

    for model_name, model in clas_models.items():
        clas_train_model(model_name, model, clasMetricLogger)

    data_Y.unique()

    st.title(" Подбор гиперпараметров для выбранных моделей. ")
    
    n_range_list = list(range(1,100,1))
    n_range_list[0] = 1

    n_range = np.array(n_range_list)
    tuned_parameters = {"n_neighbors" : n_range}
    fit_status = st.text('')

    knn_gs = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring='f1_micro')
    knn_gs.fit(data_X_train, data_y_train)
    fit_status = st.text('KNN')

    lr_grid={"C":np.logspace(-2,3,6), "penalty":["l1","l2"], "tol" : np.logspace(-4,-1,4),
                                'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
    lr_gs = GridSearchCV(LogisticRegression(), lr_grid, cv=5, scoring='f1_micro')
    lr_gs.fit(data_X_train, data_y_train)
    fit_status = st.text('LogR')

    SVC_grid=[{'kernel': ['rbf']},
                    {'kernel': ['linear']}]
    SVC_gs = GridSearchCV(SVC(), SVC_grid, cv=5, scoring='f1_micro')
    SVC_gs.fit(data_X_train, data_y_train)
    fit_status = st.text('SVC')


    tree_grid={'max_depth': np.linspace(1,10,10)} 
    tree_gs = GridSearchCV(DecisionTreeClassifier(), tree_grid, cv=5, scoring='f1_micro')
    tree_gs.fit(data_X_train, data_y_train)
    fit_status = st.text('Tree')

    rf_grid={'max_depth': list(range(1,10,1)),'n_estimators': list(range(1,30,2))} 
    rf_gs = GridSearchCV(RandomForestClassifier(), rf_grid, cv=5, scoring='f1_micro')
    rf_gs.fit(data_X_train, data_y_train)
    fit_status = st.text('RF')

    gb_gs = GridSearchCV(GradientBoostingClassifier(), rf_grid, cv=5, scoring='f1_micro')
    gb_gs.fit(data_X_train, data_y_train)
    fit_status = st.text('')

    """Рассмотрим изменение качества построенной модели на тестовой выборке в зависимости от К-соседей"""
    fig, ax = plt.subplots(ncols=1, figsize=(10,5))  
    plt.plot(n_range, knn_gs.cv_results_['mean_test_score'])
    fig.suptitle('KNN')
    fig.show()
    fig, ax = plt.subplots(ncols=1, figsize=(10,5))  
    plt.plot(np.linspace(1,10,10), tree_gs.cv_results_['mean_test_score'])
    fig.suptitle('Tree')

    clas_models_grid = {'KNN_5':KNeighborsClassifier(n_neighbors=5), 
                    str('KNN_ ' + str(knn_gs.best_params_)):knn_gs.best_estimator_,
                    'LogR': LogisticRegression(), 
                    str('LogR ' + str(lr_gs.best_params_)):lr_gs.best_estimator_,
                    'SVC':SVC(probability=True),
                    str('SVC ' + str(SVC_gs.best_params_.items())):SVC_gs.best_estimator_,
                    'Tree':DecisionTreeClassifier(),
                    str('Tree '+str(tree_gs.best_params_.items())):tree_gs.best_estimator_,
                    'RF':RandomForestClassifier(),
                    str('RF '+str(rf_gs.best_params_.items())):rf_gs.best_estimator_,
                    'GB':GradientBoostingClassifier(),
                    str('GB '+str(gb_gs.best_params_.items())):gb_gs.best_estimator_}

    for model_name, model in clas_models_grid.items():
        clas_train_model(model_name, model, clasMetricLogger)

    clas_metrics = clasMetricLogger.df['metric'].unique()

    st.title("Сравним метрики качества моделей")
    for metric in clas_metrics:
       clasMetricLogger.plot('Метрика: ' + metric, metric, figsize=(7, 6))
      

main_status = st.text('')

data_load_state = st.text('Загрузка данных...')
data = load_data()
data_load_state.text('Данные загружены!')

data_load_state = st.text('Предварительная обработка данных...')
data_X,data_Y = preprocess_data(data)
data_load_state.text('Данные обработаны!')

test_size = st.sidebar.slider("test_size", 0.1, 0.9, value = 0.3)
n_estimators = st.sidebar.slider("n_estimators", 1, 20, value=5)
n_neighbors = st.sidebar.slider("n_neighbors", 1, 20, value=5)
random_state = st.sidebar.slider("random_state", 1, 20, value=10)
max_depth = st.sidebar.slider("max_depth", 1, 10, value=4)

main_status.text('В процессе обучения...')

data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(
    data_X, data_Y,test_size=test_size, random_state=1)

# Модели
models_list = ['LogR', 'KNN_5', 'SVC', 'Tree', 'RF', 'GB']

clas_models = {'LogR': LogisticRegression(random_state=random_state), 
               'KNN_5':KNeighborsClassifier(n_neighbors=n_neighbors),
               'SVC':SVC(probability=True,random_state=random_state),
               'Tree':DecisionTreeClassifier(max_depth=max_depth,random_state=random_state),
               'RF':RandomForestClassifier(max_depth=max_depth,n_estimators = n_estimators,random_state=random_state),
               'GB':GradientBoostingClassifier(n_estimators = n_estimators,random_state=random_state)}

main_status.text('Обучено!')

metrics = [precision_score, accuracy_score, recall_score, f1_score, confusion_matrix]
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
        if name in [precision_score, recall_score, f1_score]:
            score =  name(Y_pred, data_y_test,average='micro')
        else:
            score =  name(Y_pred, data_y_test)
        scorelist[model_name]=score   
        st.text("{} - {}".format(model_name, score))
    if not scorelist:
        continue
    print(scorelist)
    if name != confusion_matrix:
        scorelist = pd.DataFrame.from_dict(scorelist, orient = 'index')
        st.bar_chart(scorelist)