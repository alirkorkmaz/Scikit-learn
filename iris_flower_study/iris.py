'''
İris Veri Seti: 150 örnekten oluşur.
Her örnek 4 özellik içerir:
Sepal length (Çanak yaprağı uzunluğu)
Sepal width (Çanak yaprağı genişliği)
Petal length (Taç yaprağı uzunluğu)
Petal width (Taç yaprağı genişliği)
'''



# scikit-learn içinde hazır datasetleri bulunmaktadır. 
# Bu setler içinde load_iris fonksiyonundan yararlanarak ufak bir proje yapacağız.
# load_iris içinde veri seti + açıklamaları + etiketleri olan bir 
# BUNCH(Sözlük gibi davranan özel bir yapı) objesi döndürür. 
from sklearn.datasets import load_iris
import pandas as pd 

# fonksiyonu çağırdık ve bir değişkene atadık
iris = load_iris()

# girdi (input) verisini temsil eder, şekli 150,4 olan bir veri. 150 örnek herbiri 4 özellikten oluşuyor.
x = iris.data
# targer (output) değeri çıkış verisini tutar. uzunluğu 150 birimdir. Etiketleri içerir. (0,1,2,3) Her bir değer bir çiçeği temsil eder. 
y = iris.target
# x değeri veriyi eğitmek için kullanılır, y değeri doğru cevapları olarak bu işleme katılır.

# X'teki değerlerin ne anlama geldiklerini belirtir. ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
fauture_names = iris.feature_names
# y'deki 0,1 gibi değelerin ne anlama geldiğini tutar. ['setosa', 'versicolor', 'virginica']
target_names = iris.target_names



# x değeri normalde numpy dizini olarak çağrılır. Pandas ile bu numpy verisini daha okunablir bir 
# hale çeviriyoruz. future_namelerini de aşağıda belirttik. 
# Flitreleme, görselleştirme, analiz işlemleri daha uygun bir yapıdadır. 
df = pd.DataFrame(x, columns= fauture_names)
print("dataFrame", df)
print(df.head())


# tüm sayısal sütünlar için temel istatistikleri çıkarır. Ortalama (mean)
# Standart sapma (std)
# Minimum, maksimum, çeyrekler (25%, 50%, 75%)
# Neden kullanırız? Verinin dağılımını, aykırı değer olup olmadığını veya normalleştirme gerekip gerekmediğini 
# anlamamıza yardımcı olur. 
print(df.describe())

# df['species'] ile yeni bir sütun ekliyoruz datamıza 
# y değeri sayısal olduğu için, target_names'i kullanarak bu sayıları kategorik isimlere çeviriyoruz. 
# 0 → 'setosa', 1 → 'versicolor', 2 → 'virginica'
df['species'] = pd.Categorical.from_codes(y, target_names)

# pd.Categorical.from_codes => Sayısal değerleri kategorik değerlere dönüştüren Pandas fonksiyonudur.
# Böylece grafik çizerken, gruplarken veya analiz yaparken kategorik işlem yapmak daha kolaydır.


# her çiçek türünden kaç tane olduğu ile iligili bilgi verir. 
# hangi sınıfın veride daha fazla veya eksik olduğu ile iligi bilgi verir.
# Tüm değerlerin aynı olması demek modelin eğitiminde bir sorun yaşanmayacağı anlamına gelir.
# Eğer dengesizik olsaydı modelin adaletsiz öğrenmesine neden olurdu.
print(df["species"].value_counts())


# bu modül datayı train ve test olarak ayırmaya olanak tanır. Bu işlemi rastgele yapar.
# rastgele veri ayrımı overfitting olmasını engeller.
from sklearn.model_selection import train_test_split

# x ve y verisetlerini test ve train olarak ayırıyor.
# bu işlemi 0.2 oranlamasıyla yapıyor. %80 train, %20 test 
# X_train, modelin öğrenmesi için kullanılacak girdiyi temsil eder.
# X_test, modelin test edileceği girdi verisini temsil eder.
# Y_train, modelin öğrenmesi için output(etiketli) değeri temsil eder.
# Y_test, modelin test edileceği çıktının değerlerini belirtir.
# Aynı sonuçları almak için random_state değeri 42 yapıldı. Eğer bir değer sabitlemezsek her defasında farklı bir sonuçla karşılaşabiliriz.
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# X_train.shape X_Test.shape ile eğitim ve test veri setlerini boyutunu kontrol ediyoruz.
print("Eğitim verisi boyutu:", X_train.shape)
print("Test verisi boyutu:", X_test.shape)


# sklearn kütüphanesi içinde modeller bulunmakta, LogisticRegression bunlardan biridir.
# LogisticRegression, sınıflandırma problemleri için kullanılan bir makine öğrenmesi algoritmasıdır.
# Adında regression geçsede sınıflandırma algoritmasıdır. 
# Özellikle ikili ve çok sınıflı sınıflandırmalar için uygundur. 
# Girdi verilerine göre bir örneğin hangi sınıfa ait olduğunu tahmin eder.
from sklearn.linear_model import LogisticRegression

# bir nesne oluşturduk ve max_iter değerini belirledik.
# kaç tane iterasyon yapması gerektiğini belirttik. 200 defa dönecek.
# ConvergenceWarning gibi bir hata alırsak değeri arttırabiliriz.
model = LogisticRegression(max_iter=200)


# X_train üzerindeki veriyi öğrenir.
# y_train, hangi örneğin hangi sınıfa ait olduğu bilgisiydi. input ve outputlarını vermiş olduk.
# .fit() ile train işlemi gerçekleştirilir.
model.fit(X_train, Y_train)
print(model.coef_) # öğrenilen katsayılar (her özelliğin ağırlığı)
print(model.intercept_) # Sabit terim


# modelin görmediği verileri kullanarak tahmin yapıyoruz. 
# X_test = Test setindeki özellikler (girdi verisi)
# model.predict() bu veriler için modelin tahmin ettiği etiketleri döndürür. 
# y_pred,test verisine ait modelin tahmin ettiği sınıf etiketini içerir.
# predict() modelin fit() ile eğitiminden sonra kullanılır. 
y_pred = model.predict(X_test)
print("Gerçek değer", Y_test)
print("Tahmin", y_pred)

# accuracy_score => Doğruluk oranını verir.
# confusion_matrix =>  Karşılık matrisi
# classification_report => Precision, recall, F1-score gibi detaylı sınıflandırma metriklerini gösterir.
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# bu fonksiyon    doğru tahmin sayısı / toplam tahmin sayısını verir
# 30 tahminin 28'i doğruysa 28/30 olur. 0.9333
accuracy = accuracy_score(Y_test, y_pred)


# her sınıf için hangi örneklerin doğru veya yanlış tahmin edildiğini gösteren 2D matris üretir.
# tahmin ve gerçek değerlerin sayılarını 3'e 3 bir matris ile gösterir. 
cm = confusion_matrix(Y_test, y_pred)

# her sınıf için detaylı performans raporu çıkartır. 
# precision: Doğru pozitif / (doğru pozitif + yanlış pozitif)
# recall: Doğru pozitif / (doğru + kaçırılanlar)
# F1 score: Precision ve recall değerlerin harmonik ortalaması 
# support: o sınıftan kaç örnek var. 
report = classification_report(Y_test, y_pred, target_names = target_names)

print(f"Dogruluk: {accuracy:.2f}\n")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)


# pipeline sınıf birden çok işi sıralı ve düzenli şekilde tek bir yapı içinde çalıştırılmasını sağlar
# özellikle veri ön işleme (standardizasyon, encoding) + model kurulumunu zincirlemek için kullanılır. 
from sklearn.pipeline import Pipeline

# Özelliklerin ölçeklendirilmesi için kullanılan StandardScaler sınıfını kullanır 
# bu işlem her özelliğin ortalamasını sıfır alır, standart sapmasını 1 yapar. 
# Bazı algortimalar farklı ölçekteki özelliklerden etkilenir..
# Bu yüzden ölçekleme işlemi modelin performansını arttırır.
from sklearn.preprocessing import StandardScaler

# ilk olarka veriyi standardize ediyoruz, sonrasında logisticregression ile eğitiyoruz
# tek bir fit ve precdict ile tüm işlemler zincirleme çalışacak. 
# Cross-validation gibi ileri konularla birlikte mükemmel uyum sağlar. 
pipeline = Pipeline([("scaler", StandardScaler()), ("logistic", LogisticRegression(max_iter=200))])



# pipeline'ın bütününün eğitir.
pipeline.fit(X_train, Y_train)

# pipelie içindeki işlemler bu sefer X_test için çalışır 
y_pred_pipeline = pipeline.predict(X_test)

print("pipeline accuracy", accuracy_score(Y_test, y_pred_pipeline))


# bu algortima bir örneğin sınıfını belirlerken en yakın komşularına bakar. 
from sklearn.neighbors import KNeighborsClassifier

# KNN nedir? 
# eğtim verisindeki her örnek , n boyutlu bir uzaydaki bir nokta gibi düşünülebilir. 
# Yeni gelen örneğin, eğitim verisindeki en yakın k adet komşusuna bakar. 
# En çok görülen sınıf neyse, yeni örnek o sınıfa atılır 
# Örneğin k 5 ise, en yakın 5 örneğe bakılır. Bu örneklerin çoğu setosa ise bu değer setosa olarak kabul edilir.


# neden ölçeklendirme önemli, knn mesafe dayalı çalışan bir algortima olduğu için özelliklerin farklı ölçeklerde çıkması 
# öğrenmeyi bozabilir.
knn_pipe = Pipeline([("scaler",StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))])

knn_pipe.fit(X_train, Y_train)

y_pred_knn = knn_pipe.predict(X_test)

print("KNN doğruluk:", accuracy_score(Y_test, y_pred_knn))

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(Y_test, y_pred_knn))
print(classification_report(Y_test, y_pred_knn, target_names=target_names))
