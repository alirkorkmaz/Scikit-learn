'''
from ile sklearn çağırıyoruz ve bunun altında bir çok py dosyası bulunmakta. 

Bunlar şöyle sıralanır;
from sklearn.   datasets  (hazır veri setleri ve veri seti yükleme )
                model sellection (eğitim test ayırımı, cross validation, grid-search)
                pipeline  ( adım adım işlem süreci oluşturma)
                compose (ColumnTransformer gibi yapılar ???)
                preprocessing (Ölçekleme, kodlama, dönüştürme)
                metrics (doğruluk, precision, racall, f1)
                linear_model (regression ve linear modeller)
                neighbors (knn, komşuluk algoritmaları)
                tree (karar ağaçları algoritmaları)
                svm (support vector machines )
                ensemble (random forest, gradient boosting)
                naive_bayes (naive bayes sınıflayıcıları )
                inspeciton (model yorumlama, permutation_importance)
                future_selection (özellik seçimi)
                impute (eksik verileri doldurma)
                utils (yardımcı fonksiyonlar)


'''


# datasets altındaki fonksiyonlar ve sınıflar 


# hazır veri setlerini yüklemek için load_* komutunu kullanabiliriz.
from sklearn.datasets import load_iris
# load_* ile veri bir bunch objesi olarak gelir (dict. gibi davranır.)

data = load_iris()
# keys ile neleri çağırdığımı görebilirim
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
print(data.keys()) 
# data.data data.target, data.frame, data.target_name
for key in data.keys():
    print(f"{key}: {type(data[key])}")
'''
data: <class 'numpy.ndarray'>
target: <class 'numpy.ndarray'>
frame: <class 'NoneType'>
target_names: <class 'numpy.ndarray'>
DESCR: <class 'str'>
feature_names: <class 'list'>
filename: <class 'str'>
data_module: <class 'str'>
'''

# data değeri numpy ile pd.DataFrame yaparak daha düzgün görüntü elde edebiliriz.
'''
import pandas as pd

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
'''
# eğer görsel veri kullanmışsak laod_digits() örnk.
'''
from sklearn.datasets import load_digits
digits = load_digits()

digits.images.shape      # (1797, 8, 8)
digits.data.shape        # (1797, 64) -- her görsel düzleştirilmiş halidir
'''




# simülasyon veri setleri oluşturmak için make_* komutunu yazabiliriz. yani yapay veri üretir. 
from sklearn.datasets import make_classification, make_blobs, make_moons
# dierkt olarak x ve y değeri döner
# veri tamamen rastgele üretilir
# parametrelerle dengesizlik, gürültü, sınıf sayısı gibi ayarlamalar yapılır.
X, y = make_classification(n_samples=500, n_features=5, n_classes=2)


# internette olan büyük verileri çekmek üzere yapılan komut
from sklearn.datasets import fetch_california_housing
# harici veri setlerini internetten indirmek için fetch_*
'''
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

indirme yapacağı kısımdan bahseder. 
fetch_california_housing(data_home="verilerim/")
'''