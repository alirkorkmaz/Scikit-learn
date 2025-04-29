'''
standardScalar: 
özelliklerin farklı büyüklükte olduğu veri setlerinde model başarısını ciddi oranda arttırabilir.
her bir özelliği standart normal dağılıma dönüştürür. ortalama sıfır olur. Standart sapma 1 olur. 

Formül : z = x - alfa/ beta 
x = orjinal değer 
alfa = sütunun ortalaması 
beta = sütünun standart sapması 
z = ölçeklenmiş değer 

Neden Kullanılır?
Çünkü bir çok makine öğrenmesi algoritması özelliklerin ölçeklerine duyarlıdır. 
ağaç tabanlı modeller dışında çalışabilir. 

örneğin bir özellik; 0 ile 1 arasında, diğeriyse 0 ile 10000 arasında olsun, büyük olan değer modelin daha çok dikkatini çeker, 
Bu durum yanıldıcı ağırlıklar veya dengesiz karar sınırları yaratabilir. 

fit(): ortalama ve std sapma hesaplar eğitim verisi üzerinde 
transform(): Veriyi dönüştürür. 
fit_transform(): Her iksini birden yapar.

x_test değeri fit edilmez çünkü test verisini ortalama ve std si kullanılmaz. Modelin görmediği test verisi
kendi başına değerlendirilmelidir. transform edilir sadece. 

StandardScaler(copy=True, with_mean=True, with_std=True)

copy: True orjinal veriyi korur, False yerinde değiştirir
with_mean: ortalama çıkartılsın mı 
with_std: td sapmaya bölünsün mü
Not: Eğer sparse (seyrek) matrislerle çalışıyorsan, with_mean=False seçilmelidir. Aksi takdirde hata alırsın.

Tree ve görüntü modellerinde genelde gerekmez. 

'''
# Standart kullanımı 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

########################################
# Eğitim ve test ayrımıyla kullanımı 

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # fit değil! sadece transform



'''
MinMaxScaler: özellikleri belli bir aralığa genellikle (0-1) arasına ölçeklendiren bir dönüş tekniğidir.
Verileri aynı ölçeğe çekmek istediğimiz durumlarda faydalıdır. Özellikle nöral ağlar ve gradyan tabanlı yöntemler. 

MinMaxScaler nedir?

x: orjinal değer
x(scaled) = x - xmin / xmax - xmin
xmin: sütündaki minimum değer 
xmax: sütündaki max değer 
xscaled: 0 ile 1 arasındaki değer

parametreleri: 
MinMaxScaler(feature_range=(0, 1), copy=True, clip=False)
feature_range: ölçeklenecek aralık
copy: True ise orj veri değişmez 
clip: true ise min max dışındaki değerleri kırpar. 

sinir ağları, görüntü işleme PCA / LDA gibi boyut indirgemede. 
Aykırı değer varsa dikkat, çok etkilenir.

MinMaxScaler uç değerlerden çok etkilenir. Çünkü max ve min değerlere göre ölçekleme yapar. 
Bu durumda RobustScaler kullanmak daha mantıklı olabilir. 

scaler = MinMaxScaler(feature_range=(-1, 1)) 
Bu şekilde verileri -1 ile +1 arasına ölçekleriz. Bazı sinir ağlarında bu tercih edilir. çünkü aktivasyon fonksiyonları bu aralıkta çalışrı. 
'''

from sklearn.preprocessing import MinMaxScaler
#temel kullanım
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# eğtim test ayrımı ile kullanım 
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Sadece transform!


'''
MaxAbsScaler: 
özellikle seyrek(sparse) veriler veya negatif pozitif değerlerin birlikte bulunduğu durumlar için uygundur. 
her özelliği -1 ile 1 arasında ölçeklendirir. her sütunun en büüyk mutlak değeri 1 olacak şekilde tüm veriler bölünür. 
Negatif değerlerde korunur. sadece aralık -1 ile 1 arasında olur. 

Neden kullanılır: 
Sparse verilerde (çok fazla sıfır içeren matrisler) diğer scaler'lar sıfır olmayan değerleri değiştirebilirken 
MaxAbsScaler bunu yapmaz, Bu da verini seyrek yapısını korur. 

Negatif ve pozitif değer içeren veriler uygundur. Sadece değerlerin mutlak büyüklüklerine bakarak dönüştürme yapar. 
yani ortalama veya standart sapma gibi istatistiklerle oynamaz. 

MaxAbsScaler(copy=True)
False ise orjinal verinin üstüne yazar. 
'''

from sklearn.preprocessing import MaxAbsScaler
import numpy as np

# Örnek veri
X = np.array([[1, -2], [2, 0], [4, 6]])

# Ölçekleyici nesnesi
scaler = MaxAbsScaler()

# Fit ve transform işlemi
X_scaled = scaler.fit_transform(X)

print(X_scaled)



'''
RobustScaler:
Özellikle aşırı uç değerlerden etkilenmeyen bir normalizasyon yapar. bu özelliği ile standartlaşma yaparken uç değerlerin
model performansını bozmasını engellemek için kullanılır. 
Veri, medyan ve IQR interquartile range 25. ve 75. yüzdelikleri arası fark) 
median(ortanca) 50. yüzdelik değer. 
IQR (q3-q1) 75. yüzdelik - 25. yüzdelik

Neden kullanılır? 
Aykırı verilerin olduğu durumlarda güvenilirdir. 
StandardScaler ve MinMaxScaler gibi yöntemler ortalama ve std kullanılır. aykrırı değerler bu istatistikleri bozar. 
RobustScaler ise ortanca ve IQR kullanır → bu yüzden uç değerlerden etkilenmez.

RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
with_centering:True ise medyanı çıkarmayı sağlar 
with_scall: True ise IQR ile bölmeyi sağlar.
quantile_range =(25.0, 75.0): IQR hesaplaması için kullanılacak değerler
copy=True orjinal veriyi değiştirmez 

veride uç noktalar varsa kullanılabilir. 
std ve ortalama gibi istatistiklerin güvenilmez olduğunu düşünüyorsak 
özellikle finansal veriler, sensör verileri, anomali tespiti gibi durumlarda kullanılabilir.
'''

from sklearn.preprocessing import RobustScaler
import numpy as np 
x = np.array([[1,100], [2,200], [3,300], [4,4000]]) # 4000 aykırı değer
x_scaled = scaler.fit_transform(x)
print(x_scaled)




'''
Normalizer: 
Her bir örneği (satırı) kendi vektör uzunluğuna göre ölçekler. yani her satırı birim uzunlukta olacak şekilde 
dönüştürür. bu işlem veriler arasındaki açısal farkları korurken büyüklük farklarını etkisiz hale getirir.

Özellikle metin madenciliği, cosine similarity tabanlı modeller, KNN, ölçekten bağımsız mesafe tabanlı algortimalar için faydalıdır. 

her satırın normu 1 olacak şekilde yeniden ölçeklenir. 

Normalizer(norm='l2', copy=True)
norm: 'l1', 'l2' veya 'max'

L2 norm = √(4² + 3²) = √(16 + 9) = √25 = 5
Normalize edilmiş hali: [4/5, 3/5] = [0.8, 0.6]

eğer satırın büyüklüğü değil yönü önemliyse, 
veriler aynı ölçekte değilse ama vektör yönleri karşılaştırılacaksa 
TF-IDF, KNN, text classification, recommendation system projelerinde
kullanılır.
'''

from sklearn.preprocessing import Normalizer
import numpy as np

X = np.array([[4, 3], [1, 2], [0, 0]])
scaler = Normalizer(norm='l2')
X_normalized = scaler.fit_transform(X)

print(X_normalized)


'''
binarizer: 

Sayıları ikili değere çevirir. 
Verilen bir eşik değere göre verileri 0 veya 1 olarak sınıflandırır. 
Özellikle sınıflandırma, özellik seçimi, metin madenciliğinde tercih edilir. 

Binarizer(threshold=0.0)
eşik değeri değiştirilebilir. 
Bu durumda:
	•	x > 0.0 → 1
	•	x <= 0.0 → 0

Belirli bir eşik değerin anlamlı olduğu veri setlerinde 
Özellikleri sadeleştirmek veya kategoriye dönüştürmek istendiğinde 
metin verileri, TF-IDF skorları gibi değerlerde küçük olanları sıfırlamak için
görsel işleme ve derin öğrenme öncesi veri temizleme işlemlerinde 

Çıktı:    
[[1 0 1]
[0 1 0]]
'''

from sklearn.preprocessing import Binarizer
import numpy as np

X = np.array([[1.5, -0.5, 2.3],
              [0.0, 1.1, -1.2]])

binarizer = Binarizer(threshold=1.0)
X_bin = binarizer.fit_transform(X)

print(X_bin)


'''
OneHotEncoder:

kategorik verileri alıp bunları sıcaklık kodlaması denilen ikili vektörlere dönüştürür.

encoder = OneHotEncoder(
    drop=None,           # 'first' → dummy variable trap için ilk kategoriyi atar
    sparse_output=False, # sparse yerine numpy array döner
    handle_unknown='error'  # 'ignore' yapılırsa yeni kategori gelince hata vermez
)
drop: ilk kategoriyi atamak istersek.
sparse_output: False; normal nuppy array verir. varsayılan true
handle_unknown: ignore; eğitimde olmayan kategori gelirse hata verir.
categories: Kategorilerin sırasını elle belirtmek için

encoder.categories_ ile hangi sütun hangi kategori ait bakılabilir. 

Neden kullanılır ? 
Makine öğrenmesi modelleri, doğrudan kategorik metinleri anlayamaz. 
Sayısal karşılık vermek yanıltıcı olabilir çünkü aralarında sıralı ilişki yoktur. 
OneHotEncoder sayesinde her kategoriye ayrı bir boyut açılır. - bağımsızlık korunur.

Ne zaman kullanmalıyız? 
Nominal (etiket değeri olan ama sıralama içermeyen değişkenlerde.) renk, ülke, marka
Karışık veri setlerinde ColumnTransformer ile birlikte
Pipeline içindeki otomatik dönüşümlerde. 


'''

import numpy as np
from sklearn.preprocessing import OneHotEncoder

X = np.array([['Kırmızı'], ['Mavi'], ['Yeşil'], ['Mavi']])

encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X)

print(X_encoded)




'''
OrdinalEncoder:

Kategorik verileri sayılarla temsil eder. Ama bu temsil sıralıdır.
x = np.array([["küçük"], ["orta"], ["büyük"], ["orta"]])
varsayılan olarak alfabetik sıralama yapar. 

Neden Kullanılır: 
Eğer kategoriler doğal bir sıralamaya sahipse sıralı temsil kullanmak mantıklıdır. 
Bazı algoritmalar bu sıralı ilişkiden faydalanabilir. karar ağaçları, XGBoost

Eğer kategorilerin sıralı bir anlamı yoksa kullanılmaz. 
'''

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

x = np.array([["küçük"], ["orta"], ["büyük"], ["orta"]])
encoder = OrdinalEncoder()
x_encoded = encoder.fit_transform(x)
print(x_encoded)


'''
LabelEncoder: 
sadece hedef değişkeni (y) yani etiketleri sayısallaştırmak için kullanılır. 
y = ["erkek", "kadın", "kadın", "erkek", "diğer"]
LabelEncoder çıktı:
[1,0,0,1,2]

sadece label değerleri sayısallaştırmak için kullanılır. 

Nerede kullanılır?
Genelde sadece hedef değişkenlerde (y) kullanılır. 
Özellikle sınıflandırmada algoritmalarında y = ["A", "B", "C"] gibi kategorik hedefler için uygundur. 
Özellik sütünlarında kullanma. Çünkü sayıların sıralı anlamı yanıltıcı olabilir. 

alternatif olarak pandas.factorize() de aynı işlemi yapar. 
'''

from sklearn.preprocessing import LabelEncoder

y = ['erkek', 'kadın', 'kadın', 'erkek', 'diğer']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(y_encoded)


'''
PolynomialFeatures: ??? çok anlamadım 

Basit verilerden karmaşık ilişkiler çıkarmak için çok güçlü bir dönüştürücüdür. 
Elimizdeki sayısal değişkenlerden, 
kendi karesi x^2
küpü X^3
iki değişkenin çarpımı x1*x2 gibi
yeni özellikler üretilir. 
Bu sayede lineer modellerle bile doğrusal olmayan ilişkiler yakalayabiliriz. 

Neden kullanılır?
Bazı verilerde bağımsız değişkenler ile hedef değişken arasındaki ilişki doğrusal değildir. 
Ama veriyi zenginleştirirsek lineer modeller bile bunu öğrenebilir. 
çıktı:
[[2. 3. 4. 6. 9.]]
Açıklama:
	•	2 → x1
	•	3 → x2
	•	4 → x1²
	•	6 → x1 * x2
	•	9 → x2²

degre: Polinom derecesi 
interaction_only: Sadece çarpım terimleri üretir. False varsayılan
include_bias: 1 sabit sütunu ekler mi? True = ekler
'''
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

x = np.array([[2,3]])
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)
print(x_poly)


'''
PowerTransform: 

Dağılımı çarpık olan verileri daha normal hale getirmek için çalışrı. 
Bu, özellikle Regresyon, LDA, PCA, GaussianNB gibi modellerde işe yarar çünkü bu modeller 
genellikle verinin normal dağıldığını varsayar. 
Neden kullanılır: 
Özelliklerin simetrik ve ortalama 0, standart sapması 1 olacak şekilde olması beklenir. 
Çarpık veri modelin performansını düşürebilir. PowerTransformer bu sorunu kökten çözer. 

method Yeo-Johnson olursa pozitif veya negatif değerlerle çalışabiliriz.
method Box-Cox olursa sadece pozitif değerlerle çalışabiliriz. 

parametreler: 
method: yeo-johnson veya box-cox
standardize: True, dönüşümden sonra ortalaması 0, std'si 1 yapılır. 
copy: Orjinal veriyi değiştirmemek için kullanılır. 
!! Box-cox kullanılacaksa verinin tüm değerleri pozitif olacaktır. 

Özelliklerin dağılımı çok çarpıksa, GaussianNB, PCA, LDA gibi normallik varsayan algoritmalar 
kullanılacaksa, 
Lineer regression yapılacaksa ama veride değişen varyans (heteroskedastiste) varsa 

'''
from sklearn.preprocessing import PowerTransformer
import numpy as np

# Simüle edilmiş verimiz
X = np.array([[1], [2], [3], [4], [5], [6], [20]])

# Transformer nesnesi oluştur
pt = PowerTransformer(method='yeo-johnson', standardize=True)

# Uygula
X_trans = pt.fit_transform(X)

print(X_trans)


'''
QuantileTransformer:

verinin kümülatif dağılım fonksiyonunu hesaplayarak, veriyi düzgün bir dağılıma, 
veya normal bir dağılıma dönüştürür.

Neden kullanılır?
Outlier (aykırı değer) varsa -> etkilerini azaltır. 
Özellikler farklı dağılımlarda olabilir. model bundan olumsuz etkilenebilir.
Veriyi benzer dağılıma sokarak modelin daha kararlı ve tutarlı sonuçlar vermesini sağlar. 

Ne yapar?
Verideki her değerin sırasına (rank) göre, o değerin persentilini hesaplar, sonra bu değeri 
hedef dağılıma karşılık gelen bir değere dönüştürür.

parametreler:
n_quantiles: CDF için kullanılacak adım sayısı (veri sayısından fazla olmamalı)
output_distribution: uniform(veriyi 0 ile 1 arasında eşit dağıtılır) 
veya normal(veriyi ortalaması 0 ve standart sapması 1 olacak şekilde normal dağıtır) kullanılır.
ignore_implicit_zeros: Özellikle sparse matrislerinde etkili
subsample: CDF tahmini için örneklem sayısı
random_state: rastgelelik kontrolü

Ne Zaman Kullanılır?
veride çarpıklık (skewness) varsa,
aykırı değerleri baskılamak istiyorsan,
tüm değişkenleri aynı dağılıma sokarak modelleri daha stabil hale getirmek istiyorsan 
Tree-based modeller hariç tüm modellerde işe yarar. (SVM, LDA, PCA)
*Sıralamaya duyarlıdır, bu yüzden çok büyük veri setlerinde n_quantiles ayarını azaltmasın 
*Eğer hedef dağılım normal ise, veriler Z-skoruna benzeyecek şekilde dönüştürülür. 
*Çok az veri varsa, dağılımı bozabilir. 

'''

from sklearn.preprocessing import QuantileTransformer
import numpy as np

# Örnek veri
X = np.array([[1.0], [2.0], [2.1], [2.2], [3.0], [20.0]])

# Nesne oluştur
qt = QuantileTransformer(output_distribution='normal', n_quantiles=100)

# Uygula
X_trans = qt.fit_transform(X)

print(X_trans)


'''
FunctionTransformer:

kendi yazdığımı herhangi bir python fonksiyonu bir veri dönüşüm nesnesine çevirir. 
Böylece bu fonksiyonu, Pipeline içinde ya da GridSearchCV ile kullanılabilir hale getirebiliriz.

Ne zaman kullanılır?
Scikit learn içinde olmayan bir dönüşüme ihtiyaç varsa,
Log dönüşümü, kare alma gibi matematiksel dönüşümleri uygulamak istiyorsak 
fit() / transform() metoduna sahip bir sınıfa ihtiyaç varsa. 

log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
parametreler: 
func: dönüştürme işlemi np.
inverse_func = Ters dönüşüm fonksiyonu
validate = True: input'un 2D olduğundan emir olur. 
kw_args = func için ekstra argümanlar
inverse_func = için ekstra argümanlar
feature_names_out = isimleri değiştirme

Fonksiyonun NumPy array alıp array döndürmesi gerek. 
Validate= True ise veri 2D olmalı. örneğin n_samples, n_features
çok karmaşık işlemler için bazen TransformerMixin ile kendi sınıfını yazmak daha mantıklıdır. 
'''

from sklearn.preprocessing import FunctionTransformer
import numpy as np

log_transformer = FunctionTransformer(func=np.log1p)
x = np.array([[1], [10], [100], [1000]])
x_trans = log_transformer.transform(x)
print(x_trans)


