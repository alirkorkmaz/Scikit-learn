from sklearn.pipeline import (
    Pipeline,
    make_pipeline
)


'''
pipeline tüm veri ön işleme ve model adımlarını tek bir yapı içinde 
tanımlanmasına olanak tanır. 

iyi bir makine öğrenmesi sisteminde pipeline kavramı yatar. 

nedir? 
veri setine sırasıyla işlemler (scaling, encoding, model vb.) uygulamak için
kullanılır. Özellikle:
ön işleme + modelleme adımlarının bireleştirilmesi sağlanır. 
fit, predict, score gibi metodları tek bir yerden yönetmeyi sağlar. 

GridSearchCV gibi yöntemlerle tüm pipeline'ı birlikte optimize eder.

'''

# Temel Kullanım:
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
pipe.fit(x_train, y_train)
pipe.predict(x_test)

# her adım bir isim (strig) ve bir işlem (transformer/ estimator) nesnesi içerir.


# make_pipeline nedir?
'''make_pipeline, pipeline işleminin daha kısa versiyodur. 
tek fark, adları otomatik oluşturur. (standardscaler, logisticregression gibi)'''

from sklearn.pipeline import make_pipeline 
pipe = make_pipeline(StandardScaler(), LogisticRegression())

# isimleri kendisi verdiği için GridSearchCV gibi yerlerde parametre 
# isim yazarken: 'logisticregression__C' gibi olur.

'''
Ne zaman kullanılabilir ? 
Verileri işleme adımlarını zincirleme uygularken. 
Modelin yanında preprocessor kullanırken, 
GridSearchCV, cross_val_score gibi işlemlerle birlikte tek parça sistem kullanmak isteniyorsa.
kod tekrarını ve karmaşıklığı önlemek istiyorsak.



'''