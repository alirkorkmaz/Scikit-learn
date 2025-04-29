from sklearn.impute import (
    SimpleImputer,
    KNNImputer
)


'''
SımpleImputer: 
veri setindeki eksik değerleri missing values doldurmak için kullanılır. 
Eksik veri doldurma işlemini (imputation) işlemini çok hızlı ve basit bir şekilde yapar. 
Verideki NaN değerleri belirli bir stratejiyle doldurur.

eksik veri varsa makine öğrenmesi modelleri çalışmaz. 
Sayısal eksik değerleri ortalama medyan veya mod ile doldurur.
Kategorik eksik verileri en sık geçen değerle doldurur. 

Eksik değer ile bulur np.nan veya belirtilen başka bir eksik değer.
berlirlenen stratejiye göre . ortalama, medyan, en sık görülen değer, sabit bir değer.

parametreler: 
missing_value: eksik veri olarak neyi kabul edeceğiz. 
strategy: doldurma yöntemi (mean, median, most_frequent, constant)
fill_value: strategy constant seçilirse kullanılır.

mean: sayısal değerlerde normal dağılım varsa,
median: sayısal değerlerde aykırı değer varsa,
most_frequent: kategorik  verilerde en çok tekrar eden değeri doldurmak için 
constant: Özelleştirilmiş sabit bir değer atamak için 

sayısal verilerde mean veya median seçerken dikkat.
Eğer outlier çoksa, median daha güvenli bir seçim olur.
Eğer eksik veri çok fazlaysa o kolu silmek daha iyi olur.

'''

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)



'''
KNNImputer:
eksik değerleri doldurmak için K-nearest Neighbors yöntemi kullanılır.

Eksik değeri doldurmak için en yakın k komşunun değerine bakar. 
ve ortalamsını alır. 

eksik değer içeren her örnek için, eksik olmayan özelliklere göre mesafe  hesaplar.
En yakın k komşuyu bulur. 
Eksik özellik bu komşuların ort doldurulur.

n_neighbors: kaç komşuya bakılacak
weights: uniform veya distance
metric: mesafe ölçüsü, 
missing_values: eksik değer olarak neyi kabul edeceğiz.

özellikler arası ilişkiyi dikkate alır. 
eksik veri doldurmada daha akıllıdır.
Özellikle benzer gruplar var ise çok iyi sonuçlar verir. 
Büyük veri setlerinde yavaş çalışabilir.
bellek kullanımı yüksek olablir. 
çok eksik veri varsa iyi çalışmayabilir. 

eğer özellikler farklı ölçeklerde ise önce mutlaka ölçeklendirilir.

weights="distance" seçersek  yakın komşulara daha fazla ağırlık verilir. daha hassas doldurma olur .

'''

from sklearn.impute import KNNImputer

# Imputer oluştur
knn_imputer = KNNImputer(n_neighbors=5, weights="uniform")

# Eğitim verisine uygula
X_train_filled = knn_imputer.fit_transform(X_train)

# Test verisine uygula
X_test_filled = knn_imputer.transform(X_test)