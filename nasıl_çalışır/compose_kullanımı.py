from sklearn.compose import (
    ColumnTransformer,
    make_column_transformer
)

'''
Veri setindeki farklı sütünlara farklı dönüşümler uygulamak için kullanılır. 

age, salary gibi sayısal değişkenlere -> StandardScaler 
gender, region, gibi kategorik değişkenlere -> onehotencoder


ColumnTransformer -- Elle yapılandırma 
'''
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["age", "salary"]),
    ("cat", OneHotEncoder(), ["gender", "region"])
])

# işlem adı, uygulanacak işlem ve hangi sütunların uygulanacağı bilgilerini alır.
# Sayısal sütünlar scale edilir, kategorik sutunlar one-hot encode edilir. 



# make_column_transform - Kısa yol 
from sklearn.compose import make_column_transformer

preprocessor = make_column_transformer(
    (StandardScaler(), ["age", "salary"]),
    (OneHotEncoder(), ["gender", "region"])
)

'''
Aynı işlevi yapar fakat adları otomatik verir. 
Bu nedenle GridSearchCV gibi yerlerde adlar biraz farklı olur.
onehotencoder__handle_unknown gibi


Neden önemli!

Modern veri setlerinde hem sayısal hem kategorik veri vardır. 
Bir pipeline içinde farklı sütunlara farklı bir işlem uygulamak için en iyi yöntemdir.
'''