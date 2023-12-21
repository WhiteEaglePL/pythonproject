import pandas as pd  # Importowanie biblioteki pandas i nadanie jej aliasu 'pd'
import numpy as np  # Importowanie biblioteki numpy i nadanie jej aliasu 'np'
from sklearn.feature_selection import RFE, RFECV  # Importowanie klas RFE i RFECV z modułu sklearn.feature_selection
from sklearn.svm import SVC  # Importowanie klasy SVC (Support Vector Classifier) z modułu sklearn.svm
from sklearn.preprocessing import LabelEncoder, StandardScaler,MinMaxScaler  # Importowanie klas LabelEncoder, StandardScaler, MinMaxScaler z modułu sklearn.preprocessing
from sklearn.metrics import accuracy_score, recall_score  # Importowanie funkcji accuracy_score i recall_score z modułu sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier  # Importowanie klasy KNeighborsClassifier z modułu sklearn.neighbors
from warnings import simplefilter  # Importowanie funkcji simplefilter z modułu warnings
import warnings  # Importowanie modułu warnings
import random  # Importowanie modułu random
from sklearn.model_selection import StratifiedKFold
from aggregationslib.aggregation import arithmetic, median
from scipy.stats import gmean
import time

simplefilter(action='ignore', category=FutureWarning)  # Ignorowanie ostrzeżeń typu FutureWarning

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")  # Ignorowanie wszystkich ostrzeżeń

# Ścieżki do plików CSV
data_path = "ovarian_61902DATA.csv"
labels_path = "ovarian_61902LABELS.csv"

# Tworzenie nazw kolumn dla danych
column_names = [f"Attribute{i}" for i in range(1, 253)]

# Odczyt danych z pliku CSV i nadanie nazw kolumn
df = pd.read_csv(data_path, names=column_names)
cols = ["class"]
# Odczyt etykiet z pliku CSV
df_labels = pd.read_csv(labels_path, names=cols)

# Zmiana indeksów w ramce danych na bardziej czytelne etykiety 'Object1', 'Object2', itd.
for i in range(1, len(df) + 1):
    df = df.rename(index={i - 1: f"Object{i}"})

# Wyodrębnienie danych jako macierzy X
X = df.values
# Wyodrębnienie etykiet jako wektora y
y = df_labels['class'].values

# Inicjalizacja kodera etykiet
le = LabelEncoder()
# Zakodowanie etykiet jako liczb całkowitych
y = le.fit_transform(y)

# Lista etykiet dla różnych metod tasowania danych
shuffle_labels = ["shuffle return", "shuffle without return", "array split"]


# Definicja funkcji do tasowania danych z powtórzeniami i tworzenia podtabel
def shuffleReturn(arr, num_subarrays):
    arr_length = len(arr)
    random.seed(time.time())
    subarrays = []
    values_per_subarray = arr_length // num_subarrays
    for _ in range(num_subarrays):
        subarray = random.choices(arr, k=values_per_subarray)
        subarrays.append(subarray)
    return subarrays


# Definicja funkcji do tasowania danych bez powtórzeń
def shuffleWithoutReturn(array, subtable_size):
    if subtable_size > len(array):
        raise ValueError("Rozmiar podtabeli nie może być większy niż długość tablicy.")

    indeksy = random.sample(range(len(array)), len(array))
    dlugosc_podtabeli = len(array) // subtable_size
    reszta = len(array) % subtable_size

    podtabele = [array[indeksy[i * dlugosc_podtabeli:(i + 1) * dlugosc_podtabeli]] for i in
                 range(subtable_size)]

    if reszta:
        ostatnia_podtabela = [array[indeksy[-reszta:]]]
        podtabele += ostatnia_podtabela

    return podtabele


svc = SVC(kernel='linear')

# Lista różnych wartości dla 's' (liczba podtabel)
s_values = [5, 10, 20, 50, 100]

# Lista różnych wartości dla 'k' (liczba sąsiadów w algorytmie k-najbliższych sąsiadów)
k_values = [3, 5, 7, 9, 11]

# Pusta lista do przechowywania wyników
result_data = []

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


scaler = StandardScaler()
rfe = RFE(estimator=svc, step=1, n_features_to_select=200)


class Experiment:

    def __init__(self, s, k, AUC, stdAUC, AUCGEO, stdGEO,  AUCMEAN, stdMEDIAN, splitMethod):
        self.s = s
        self.k = k
        self.AUC = [AUC]
        self.stdAUC = 0
        self.AUCGEO = [AUCGEO]
        self.stdGEO = 0
        self.AUCMEAN = [AUCMEAN]
        self.stdMEDIAN = 0
        self.splitMethod = splitMethod


experiments = []

for s in [5, 10, 20, 50, 100]:

    for k in [3, 5, 7, 9, 11]:
        experiments.append(Experiment(s, k, None, 0, None, 0, None, 0, "arraySplit"))

    for k in [3, 5, 7, 9, 11]:
        experiments.append(Experiment(s, k, None, 0, None, 0, None, 0, "shuffleReturnn"))

    for k in [3, 5, 7, 9, 11]:
        experiments.append(Experiment(s, k, None, 0, None, 0, None, 0, "shuffleWithoutReturnn"))



for train_index, test_index in kfold.split(X, y):


    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Zastosowanie skalowania do zbioru treningowego

    X_train_scaled = scaler.fit_transform(X_train)

    # Zastosowanie skalowania do zbioru testowego
    X_test_scaled = scaler.transform(X_test)

    rfe.fit(X_train_scaled, y_train)
    selected_feature_names = np.array(column_names)[rfe.get_support()]



    for s in s_values:

        sumAccuracies2Final = []
        sumAccuraciesGeometricMeanFinal = []
        sumAccuraciesMeanFinal = []

        for i in range(3):

            if i == 0:

                # print(f's = {s}, k = {k}')
                subtables = shuffleReturn(selected_feature_names, s)
                splitMethod = "shuffleReturnn"

            elif i == 1:

                subtables = shuffleWithoutReturn(selected_feature_names, s)
                splitMethod = "shuffleWithoutReturnn"

            elif i == 2:

                subtables = np.array_split(selected_feature_names, s)
                splitMethod = "arraySplit"

            # print(s)
            # arraySplit = np.array_split(selected_feature_names, s)
            # subtables = arraySplit

            sumAccuraciesArithmetic = []
            sumAccuraciesGeometricMean = []
            sumAccuraciesMean = []

            # print(f"Number of subtables {s}, name of sort: {splitMethod}")

            for k in k_values:

                sumAccuracies1 = []
                sumAccuracies3 = []

                tempValue = 0

                # print(f"s = {s}, k = {k} split method: {splitMethod}")
                # if k == 11:
                #     print("==================")

                predictions = []
                for subtable in subtables:

                    columnIndexes = []
                    test = X_train
                    temp = []

                    for feature_name in subtable:

                        # Czyszczenie SubtableValues

                        column_index = column_names.index(feature_name)
                        columnIndexes.append(column_index)

                        values = X_train_scaled[:, column_index]


                    X_subtable = X_train_scaled[:, columnIndexes]
                    X_subtable_test = X_test_scaled[:, columnIndexes]
                    # print("test")

                    # Trenowanie modelu KNN
                    knn = KNeighborsClassifier(n_neighbors=k)
                    knn.fit(X_subtable, y_train)

                    y_pred = knn.predict(X_subtable_test)
                    accuracy = accuracy_score(y_test, y_pred)

                    # print(accuracy)
                    predictions.append(y_pred)
                    sumAccuracies1.append(accuracy)

                arithmeticMean = arithmetic(sumAccuracies1)
                geometricMean = gmean(sumAccuracies1)
                medianMean = median(sumAccuracies1)

                # if s == 5 and k == 3 and splitMethod=="arraySplit":
                #     # print(testowy)
                #     print(predictions)
                #     print(sumAccuracies1)
                #     print(mean_accuracy1)
                #     print(geometricMean)
                #     print(medianMean)
                #     print(f"k = {k}, s = {s}")
                #     print("===================================================")

                sumAccuraciesArithmetic.append(arithmeticMean)
                sumAccuraciesGeometricMean.append(geometricMean)
                sumAccuraciesMean.append(medianMean)

                if len(sumAccuraciesArithmetic) == 5:
                    print(sumAccuraciesArithmetic)

                for experiment in experiments:
                    if (experiment.s == s and experiment.k == k and experiment.splitMethod == splitMethod):

                        for accuracy in sumAccuraciesArithmetic:
                            experiment.AUC.append(accuracy)

                        for accuracy in sumAccuraciesGeometricMean:
                            experiment.AUCGEO.append(accuracy)


                        for accuracy in sumAccuraciesMean:
                            experiment.AUCMEAN.append(accuracy)

for experiment in experiments:
    experiment.AUC.remove(None)
    experiment.AUCGEO.remove(None)
    experiment.AUCMEAN.remove(None)

    result_data.append(
        [experiment.s, experiment.k, arithmetic(experiment.AUC), np.std(experiment.AUC), arithmetic(experiment.AUCGEO), np.std(experiment.AUCGEO), arithmetic(experiment.AUCMEAN), np.std(experiment.AUCMEAN), experiment.splitMethod])

import pandas as pd

result_df = pd.DataFrame(result_data, columns=["s", "k", "AUC (Arithmetic)", "STDDEV (Arithmetic)", "AUC (Geometry)", "STDDEV (Geometry)", "AUC (Median)", "STDDEV (Median)", "Split Method"])

# print(result_df)
result_df.to_excel("wyniki.xlsx", index=False)

df = pd.read_excel("wyniki.xlsx")

df = df.sort_values(by=["Split Method", "s"])

df.to_excel("wyniki.xlsx", index=False)
