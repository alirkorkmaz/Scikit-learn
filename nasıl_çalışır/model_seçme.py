
from sklearn.datasets import load_diabetes

data = load_diabetes()
x = data.data
y = data.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
# 	•	random_state: Aynı bölmeyi tekrar üretmek için sabit sayı


# cross_val_score

# veriyi k-fold gibi yöntemlerle böler ve bu modeller üzerinde test yapılır.

# cv değeri kat sayısı veya çapraz doğrulama değeridir. 
# hangi metriği kullanacağını belirtir
'''
çapraz doğrulama: eğitim verisini ezber yapmadan, gerçek hayatta nasıl performans göstereceğini ölçmek için kullanılır.
k-fold Cross validation: veri seti cv değerindeki sayı kadar parçalara bölünür. genelde 5 veya 10
her seferinde bir parça test seti kalanı train seti olur.
bu işlem k kez tekrarlanır.
her seferinde model yeniden eğitilir ve test edilir. 
Sonuçta k adet doğruluk/f1/vs gibi score değerleri bulunur.
Amaç her parçalı veri bir kez test veri seti ve k-1 kez eğitim seti olmuştur. modelin daha tutarlı davranması amaçlanır.

verinin dağılımı dengesiz ise tek test veri seti yanıltıcı olabileceği için bu yöntem kullanılır.

Scoring parametresi neden seçilir?
Modelin neye göre iyi olduğunu belirleyen ölçüttür. 
Modelin başarısını ölçmek için bir kriter seçmiş oluruz. 

Model seçimi neden fark yaratabilir? 

Degesiz veri seti varsa accuracy değeri yüksek olsa bile model işe yaramaz olabilir. 
Tahminlerde dikkatli olunması gereken yerlerde presicion, recall, f1 daha uygun olabilir.

Doğru sınıflama kadar, yanlış sınıflamanın maliyetide önemli ise, f1 veya özel metrikler kullanılır.

Örnek olarak bir fraud detection modelimiz var, %99 accuracy alıyoruz, verilerin %99 u fraud değil.
ama fraudları hiç bulamıyoruz, Bu modelin gerçekte işe yaramadığını gösterir. recall değeri yüksekse mesela
fraudları yakaladığımız anlamına gelir ve bu değerli bir bilgidir.

Hangi modeli neden kullanmalıyım ? 

Problem türü tanımlaması yapmak önemlidir. 
dengesiz sınıflar var ise, %95 hayır %5 evet gibi, (presicion, recall, f1, roc_auc) kullanılır.
eşit sayıda sınıf var ise, %50 evet, %50 hayır gibi (accuracy, f1) kullanılır.
hataların maliyeti farklı ise yanlış negatif çok tehlikeli, (recall önce yaka )
yanlış pozitif tehlikeli, yanlış kişiye kredi vermek (precision)
regresyon modelleri, sürekli değer tahmini (fiyat, sıcaklık) (r2, neg_mean_squared_error, neg_mean_absolute_error)


Problem: Sınıflandırma mı?
├── Dengeli mi?
│   └── Evet → accuracy / f1
│   └── Hayır → recall / precision / f1 / roc_auc
└── Regresyon mu?
    ├── Büyük hatalar mı önemli? → mse
    ├── Ortalama hata mı önemli? → mae
    └── Açıklama oranı mı önemli? → r2

öneri olarka modeli eğitirken farklı kombinasyonlar kullanarak hangisinin verim için mantıklı olan metrik olduğunu öğrenebilirim.
'''
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x, y, cv=5, scoring="accuracy")
print(scores.mean())



'''
bir modelin, tüm veri seti üzerinde çapraz doğrulama yaparak tahmin ettiği sonuçları döndürür. yani 
her bir örnek, eğitim sırasında görülmemiş bir fold a denk getirilir ve bu fold'daki test verileri için tahmin yapılır.
Sonuçta modelin gerçek hayatta nasıl tahmin yapacağını daha yakın bir tahmin çıktısı elde edilir. 

Her bir veri noktasının eğitim sırasında görülmeden tahmin edilmesini sağlar. 
Böylece overfitting riski olmadan tahmin sonuçları elde edilir.

Diyelim ki 100 örnek var, cv=5 yaptın.
	•	cross_val_score: “Fold 1 → accuracy: 0.92, Fold 2 → 0.89…” gibi skorlar döner.
	•	cross_val_predict: “Veri[0] için 1, Veri[1] için 0, Veri[2] için 1…” gibi tahminler döner. Bunları confusion_matrix vs için kullanabilirsin.

Neden kullanılır: 
modeli henüz tam eğitmeden confusion matrix üretmek, ROC ve precision-recall eğrileri çizmek 
Amaç: Modelin tüm veri üzerindeki tahminlerini almak, 	•	Ne döner?: Her gözlem için tahmin (y_pred)
Tüm veri için çapraz doğrulama tahminleri döner. 
her örneğin tahmin sonuçlarını elde etmiş oluruz. 
Eğer “şu gözleme model ne tahmin etti” sorusunu soruyorsan,
cross_val_predict’i kullanırsın.

Eğer “model ne kadar başarılı” diye bakıyorsan,
cross_val_score daha doğru olur.
'''

from sklearn.model_selection import cross_val_predict
scores = cross_val_predict()



'''
scikit-learn da model parametre ayarı yapmak için kullanılır.
hyperparmeter tuning olarak geçer. bir modelin hiperparametrelerini denemek için kullandığımız bir araçtır.
model parametrelerini değiştirerek en iyi kombinasyonu elde etmemiz gere. 
Modelin hiperparametrelerini farklı kombinasyonlarla çapraz doğrulama yaparak test eder. 
en iyi sonucu veren ayarları bulur. 
Parametre olarak; bir model alır, denenecek parametreler bir dict veya list olabilir, hangi metric ile
başarı oranı belirlenecek bu yüzden bir metric değeri alır, cv ile cross validation değeri alır, eğitim aşamasında çıktı istiyorsa
ve paralel işlem sayısı parametrelerini alır. 

Ne zaman kullanılır: 
KNN, SVC, RandomForest, LogisticRegression, GradientBoosting gibi modellerin parametrelerini optimize etmek için
elimizde doğrulama verisi yoksa (yada overfitting'den kaçınmak istiyorsak)
ROC, recall, precision gibi metriklere göre ayar yapmak için 

Kullanılan modele göre parametre değişmektedir.

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
print(model.get_params())
Her modelin parametresini bu şekilde çekebiliriz.
param_grid = {
    'C': [0.1, 1, 10],
    ...
}
böyle bir girdide c değeri için 3 farklı değer denemiş olacağız.
'''
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
param_grid = {'C': [0.1, 1, 10]}
x = 5
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid.fit(x, y)
print(grid.best_params_)





'''
girdsearchcv yapısına çok benzer daha hızlı ve daha esnektir. belirtilen hiperparametre dağılımından belirli sayıda rastgele örnek alır.
Her örnek için modeli eğitir ve performansını değerlendirir. 
Daha az kombinasyonla hızlıca iyi souçlar bulmak için kullanılır. 
parametre olarak model, model içindeki parametrelerin kurgusu, kaç farklı kombinasyon denenecek, hedef metrik, kaç katmanlı cross validation olacak
random_state değeri(her zmana aynı verilerle bölünsün diye) ve iş parçacığı olarak parametreler alır.

parametre kombinasyonları çok fazlaysa ve hızlı bir sonuç isteniyorsa kullanılabilir. çok hassas ayarlama
gerekiyorsa ilk olarak random kullanılır parametreler daraltırlır ve grid ile tekrardan eğitim yapılır.

Sadece sabit listelerle değil dağılımlarla da çalışır, bu sayede çok geniş değerler üzerinde örnekleme yapılabilir.

uniform (loc, scale) loc:başlangıc scale: aralık 
randint (low, high) tam sayı aralığı 
loguniform (a, b) logaritmik olarak dağılmış değerler (çok yaygın)
from scipy.stats import loguniform

param_dist = {
    'C': loguniform(0.01, 100),  # Daha iyi arama için
}

'''
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

param_dist = {'C': uniform(0.1, 10)}
search = RandomizedSearchCV(LogisticRegression(), param_dist, n_iter=10, cv=5)
search.fit(X, y)




'''
veri kümesini k eşit parçaya böler, normal cross validation yapısındadır. sınıf dağılımı dengesiz ise 
kullanılması tavsiye edilmez. 

** veri noktası kavramı bir veri kümesindeki her biri satırı temsil etmektedir.
eğer cv= 5 ise 10 veri noktan varsa, 10 değeri 5 e bölünür. 0-1, 2-3, 4-5 ... şeklinde palaşılır
n_splits: cv değeri (kaç parçaya bölünecek)
shuffle: veiriyi karıştırmak istersen true 
random_state
'''

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold

model = LogisticRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print("Doğruluk Ortalaması:", scores.mean())





'''
StratifiedKFold, veriyi bölerken her fold'da sınıf dağılımını korumaya çalışır, veride dengesiz sınıflar var ise 
her train test grubunda bu oranı sabit tutmaya çalışır. 
bir sınıflandırma problemi çözüyorsak, dengesizlik varsa kullanılabilir. Regression problemlerinde gerkli değildir. 

Dikkat: skf.split(X, y) → hem X hem y verilmelidir. Çünkü StratifiedKFold, y’yi dikkate alarak bölme yapar.
'''

from sklearn.model_selection import StratifiedKFold
import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 5 tane 0, 5 tane 1

skf = StratifiedKFold(n_splits=2)

for train_index, test_index in skf.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    print("Train labels:", y[train_index], "Test labels:", y[test_index])





'''
Veriyi zaman sırasına göre artan şekliyle böyler. Yani geçmiş verilerle modeli eğitir, yeni verilerle
modeli test eder. 
finans, sağlık, hava durumu gibi zamana bağlı verilerde kullanılır. 
TimeSeriesSplit(n_splits=3) ile:
	•	1. Fold:
	•	Train → [1, 2, 3, 4]
	•	Test  → [5, 6]
	•	2. Fold:
	•	Train → [1, 2, 3, 4, 5, 6]
	•	Test  → [7, 8]
	•	3. Fold:
	•	Train → [1, 2, 3, 4, 5, 6, 7, 8]
	•	Test  → [9, 10]

    
Train veri seti her seferinde genişliyor, asla geriye gitmiyor. 
test veri seti her seferinde daha sonraki zamanı kapsıyor. 

neden diğerlerinden farklı?
Diyelimki zaman serisi olsun verimizde, borsa fiyatları olsun elimizde, klasik kfold kullanırsak belkide 
gelecekteki verilerle geçmişi tahmin etmiş olacağız, bu veri sızıntısı anlamına gelir, (data leakage) olur modeli hatalı değerlendirir.
n_splits: kaç adet flod oluşturulacak
max_train_size: eğitim max uzunluğunu sınırlayabiliriz. 
test_size: test set uzunluğunu kontrol edebiliriz. 
'''
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
tscv = TimeSeriesSplit(n_splits=3)

for train_index, test_index in tscv.split(X):
    print("Train:", train_index, "Test:", test_index)






'''
klasik kfold'dan daha esnek ve rastgele versiyonudur. Özellikle büyük veri setlerinde veya daha esnek bölme ihtiyaçlarında tercih edilir. 
Veriyi rastgele olarak eğitim ve test kümelerine belirli oranlarla böler, bu işlemi birden fazla kez tekrarlar. 
Her flodda rastgele train ve test ayrımı yapar, istediğimiz kadar bu işlemden yapabilri.z
 
overfitting riskine karşı modeli daha rastgele durumlara karşı test etmek için. 
modelin kararlılığını görmek için 

ShuffleSplit, zaman serisi verilerde kullanılmamalıdır, çünkü rastgele karışım (shuffle) 
geçmiş ve gelecek dengesini bozar → data leakage olur.
'''

from sklearn.model_selection import ShuffleSplit

X = np.arange(10)  # Örnek veri
ss = ShuffleSplit(n_splits=3, test_size=0.3, train_size=0.7, random_state=42)

for train_index, test_index in ss.split(X):
    print("Train:", train_index, "Test:", test_index)




'''
Bu yöntem, ShuffleSplit ile StratifiedKFold’un güzel bir birleşimi gibi düşünebilirsin.


veriyi rastgele ama sınıf oranlarını koruyarak eğitim ve test setlerine böler. 
Yani sınıflar orantılı bir şekilde hem eğitim hem test kümelerine dağılır. dengesiz (imbalanced) veri setlerinde 
kullanımı önemlidir. 
'''
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)

for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)





'''
Her bir veri noktasını ayrı ayrı test seti olarak kullanır. kalan tüm verileri eğitim seti yapar. 
n tane model eğitilir. her seferinde sadece bir örnek test olarak ayrılır, n-1 model eğitilir. 
her örnek tam olarak 1 kez test setinde yer alır. 

model n kez eğitilir. çok hassas ölçüm yapılır. overfitting riski düşük. Küçük veri setlerinde kullanılır. 
 Büyük veri setlerinde: Çünkü her örnek için bir model eğitmek zaman alır (örneğin 10.000 veri varsa → 10.000 model).
  Nerede kullanılır?
	•	Biyoistatistik
	•	Medikal tahmin
	•	Küçük örneklemli psikoloji deneyleri
	•	Finansal zaman serisi modelleme (kısıtlı veriyle)
'''
from sklearn.model_selection import LeaveOneOut
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

loo = LeaveOneOut()

for train_index, test_index in loo.split(X):
    print("Train:", train_index, "Test:", test_index)
    # X_train, X_test = X[train_index], X[test_index]






'''
Süper, şimdi de daha genel ama nadiren kullanılan bir teknikle devam edelim:

🔄 LeavePOut Nedir?

Leave-P-Out (LPO), her seferinde tam olarak p veri noktası test setine, 
kalan veriler eğitim setine konur. Bu işlem, tüm p kombinasyonları için tekrar edilir.
Eğer elimizde n örnek varsa, her defasında p tanesini test için ayırıyoruz.
Bu durumda kaç kombinasyon olacağını tahmin edebilir misin?

 C(n, p) farklı eğitim/test bölmesi oluşur (kombinasyon sayısı).

Örnek:
n = 5, p = 2 ⇒ C(5, 2) = 10 farklı bölme olur.
Küçük ve hassas veri setlerinde
✅ Model performansını en geniş kombinasyonlarla test etmek gerektiğinde
❌ Büyük veri setlerinde (kombinasyon sayısı hızla patlar!)

'''
from sklearn.model_selection import LeavePOut
import numpy as np

X = np.array([[1], [2], [3], [4]])
y = np.array([1, 2, 3, 4])

lpo = LeavePOut(p=2)

for train_index, test_index in lpo.split(X):
    print("Train:", train_index, "Test:", test_index)





'''
verileri gruplara göre bölen ve aynı gruplar içindeki örneklerin hem eğitimde hem testte aynı anda 
yer almamasını garanti eden bir çapraz doğrulama tekniğidir. 

eğer aynı kişiye ait birden fazla gözlem varsa (hastalık ölçümleri, alışveriş. )
aynı müşterinin farklı zamanlardaki haraketileri varsa, 
aynı sensör cihazından alınan veri varsa, 
aynı ürün kullanıcı veya kategoriye ait alt veri noktaları varsa 
Amaç; aynı gruba ait veriler aynı katmanda kalmalı hem eğitim hem testte bulunmalı. 

aynı değeri taşıyan örnekler birlikte tutulur. 
zorunlu olarak grups argümanı verilir. 
sınıf dağılımına dikkat etmez, sadece grup dağılımına bakar. 
Tüm gruplar her flodda ya test ya da eğitim setinde yer alır. 
diyelim ki 100 hasta var, ve her birinden 5 farklı kan örneği alındı Bu durumda grups dizisi şöyle olur.
groups = np.repeat(np.arange(100), 5)  # Her hasta bir grup

normal kFlod kullanmak kaçağa sebep olur. aynı hastanın verisi hem train hem test'e gidebilir. 
'''

from sklearn.model_selection import GroupKFold
import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([1, 1, 1, 0, 0, 0])
groups = np.array([1, 1, 2, 2, 3, 3])  # Grup numaraları

gkf = GroupKFold(n_splits=3)

for train_idx, test_idx in gkf.split(X, y, groups):
    print("Train:", train_idx, "Test:", test_idx)
