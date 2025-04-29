from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors
)


'''
KNeighborsClassifier:
Kısaca KNN algoritmasıdır.
En yakın komşulara bakarak bir öğrneğin sınıfını tahmin eder.
hiç model eğitmez. Sadece veriyi saklar ve tahmin zamanında karar verir.

Nasıl çalışır?
Yeni bir veri noktası geldiğinde(bir müşteri)
eğitim verisindeki tüm noktalarla uzaklık hesaplanır
En küçük mesafeye sahip k adet komşu seçilir. 
Bu komşuların sınıflarına bakılır.
Hangi sınıf çoğunluktaysa o sınıf seçilir.

Kullanılan mesafe ölçüsü:
Euclidean Distance(Öklid mesafesi):
Normal düz çizgi mesafesi.
Manhattan Distance, Minkowski Distance, Cosine Distance gibi uzaklık ölçme tipleride vardır.


parametreler:
n_neighbors: Kaç komşu bakılacak
weights: komşuların ağırlıklandırılması; uniform:hepsi eşit, distance:yakınlar daha etkili
metric: mefafe ölçüsü; minkowski -> euclidean

p: mesafe ölçüsünde güç parametresi;p=2 -> euclidean, p=1 -> manhattan
algorithm: Komşu arama algoritması ; auto, ball_tree, kd_tree, brute
avantajları:
çok basit ve sezgiseldir. Model eğitimi yok, sınıf sınırlarını iyi öğren, non-linear karar sınırlarını çizebilir. 
None linearkara sınırları çizebilir.

dezavantajları: 
büyük veri setlerinde yavaş. 
Özellik sayısı artınca performans düşer.
özellikleri iyi ölçeklendirmek gerekir

k değeri nasıl seçilebilir? 
küçük seçilirse -> model daha hassas olur fakat overfitting riski artar
büyük seçilirse -> model daha genelleştirilebilir, underfitting riski vardır.
k değeri kök içinde n_samples değeri ile eşit olabilir 
veya gridsearch ile en iyi k değeri aranır. 

Kullanmadan önce StandardScaler ile veri ölçeklemesi yapılmalı, 
eğer veri seti büyükse algorithm="kd_tree" veya ball_tree ile 
hesaplamalar hızlanabilir.

'''

'''
KNeighborsRegressor: 
KNN algoritmasını kullanarak sürekli değerler tahmin eder.
Sınıflandırma değil, artık regression yapıyoruz.
Yeni bir veri geldiğinde;
k en yakın komşuyu bulur
Bu komşuların çıktığı değerlerinin ortalamasnını alır.
Sürekli bir değer tahmini gerçekleşir.

Yeni bir değer geldiğinde, eğitim verisindeki tüm noktalarla mesafe hesaplar.
En küçük mesafeye sahip k komşu seçilir. 
bir k komşunun hedef değerlerinin ortalaması alınır.
	•	Özellikleri mutlaka ölçekle (StandardScaler gibi).
	•	weights='distance' seçeneğini dene → Yakın komşular daha fazla etki eder ➔ Daha doğru tahminler olabilir.

'''



'''
NearestNeighbors:
KNN tahmini yapmak için değil sadece en yakın komşuları bulmak için kullanılır.
Yani etiket tahmini yapmaz. kendi başına bir tahmin modeli değil, yakınlık sorgulayıcı.

en yakın komşuları bulmak ve kümeleme öncesi analiz etmek.

anamoli tespiti, öneri sistemi, veri noktalarının analizi. 

Verileri saklar, sorgu verisi geldiğinde mesafeleri ölçer. 
İstenen kadar en yakın komşuyu bulur.

parametreler:
n_neighbors; kaç yakın komşu aranacak
metric; mesafe ölçüsü(auto, euclidean, manhattan, cosine)
algorithm; arama yöntemi (auto, kd_tree, ball_tree, brute)
leaf_size; ağaca dayalı algoritmalarda performans optimizasyonu için kullanılır
'''