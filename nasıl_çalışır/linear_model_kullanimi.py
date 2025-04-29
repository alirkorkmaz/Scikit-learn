
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
    SGDClassifier,
    SGDRegressor,
    Perceptron,
    RidgeClassifier,
    PassiveAggressiveClassifier
)


'''
LinearRegression:
bağımsız değişkenler ile bağımlı değişkenler arasındaki doğrusal ilişkiyi modellemek.
Bir veya daha fazla girdiden sürekli bir değer tahmin etmek. 
Bir doğru, veri noktalarına en iyi şekilde uyacak şekilde çizilmiştr.

numerik verilerle kullanılır. kategorik değerlerde sayısallaştırılabilir.
özellikler arasında doğrusal ilişki olmalı. Yani özellik değeri arttıkça hedefte artmalı veya tam tersi olabilir.
özellikler ve hedefler arasında çok karmaşık ve doğrusal olamayan bir ilişki varsa başka modeller kullanılabilir.

parametreler:
fit_intercept:True; bias(b) terimini öğrenme sürecine katılmasını doğrular.
normalize="deprecated" Eski sürümlerde normalize işini yapan parametre.
model özellikleri:
model.coef_ = Ağırlıklar (her özellik için katsayılar)
model.intercept_ = Bias terimi

Güçlü Yönleri: 
Basirt ve yorumlanabilir. Hızlı çalışır, hesaplama ucuzdur. çok boyutlu verilerle de çalışabilir.
Teorisi iyi oturmuştur.

Zayıf Yanları:
sadece doğrusal ilişkileri modeller.
Aykırı değerlerden kolay etkilenir.
Özellikler arasında korelasyon varsa model bozulabilir.
x_train ve y_train değerlerini alır.
'''


'''
Ridge Regression:
lineer regression denklemine bir penalty(ceza) ekler. 
Ağırlıkları küçülterek modeli sadeleştirir. 
Özellikler arasında yüksek korelasyon varsa. 
Eğitim hatası çok düşük ama test hatası çok yüksekse. (overfitting)
Özellik sayısı çok fazla ama örnek sayısı azsa. 

parametre:
alpha: küçük alpha değeri Linear regression çok yakındır.
büyük alpha değeri ise model daha düzleşir, bazı ağırlıklar sıfıra yaklaşır.
Amaç overfitting'i azaltmak. 
'''
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=0.1)
ridge_model.fit(x_train,y_train)


'''
Lasso Regression: 
Gereksiz özelliklerin sıfır olduğu model.
Ağırlıkları sıfırlayarak gerçekten önemli olan özellikleri seçer.

Bu yüzden aynı zamanda bir Feature Selection(Özellik seçimi) yöntemidir.

Linear Regression'ın hata fonksiyonuna L1 normu ekler.

Ridge regression, ağırlıkların karelerini topluyordu.
Lesso Regression, ağırlıkların mutlak değerlerini topluyor.

Bazı ağırlıklar tam sıfır olur. 
Özelliklerin çok fazla olduğu ve bazılarının gereksiz olduğu durumlarda.
Modeli sadeleştirmek ve yorumlamayı kolaylaştırmak isteniyorsa. 
Feature selection yapmak isteniyorsa. 
Burada da alpha parametresi vardır.

Çok boyutlu veri ve az örnek olduğunda Lasso mükemmel performans gösterir. 
Özellikler çok karole ise (birbirine çok bağlıysa) Lasso hangisini sıfırlayıp hangisini tutacağına rastgele karar verir. 
Bu durumda elasticNet daha verimli olur.


en iyi alpha değerini bulmak için lassoCv kullanılır. Cross-validation yapar.
from sklearn.linear_model import LassoCV
lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_train, y_train)
print("En iyi alpha:", lasso_cv.alpha_)
'''


'''
ElasticNet Regression:

Ridge + Lasso değerlerini birleştirerek en iyi ikisini aynı anda kullanır. 
Ridge gibi ağırlıkları küçültür, lasso gibi bazılarını sıfırlayabilir.

l1_ratio değeri: 1 e yakınlaştıkça lasso özelliklerinden faydalanma olasılığı daha yuksektir. 
0 a yaklaştıkça ridge özelliklerinden faydalanma artar. 0.5 de ise %50 %50 oranınıdadır. 

Özellik sayısı çok fazla ise, Özellikler arasında çok fazla korelasyon var ise, FeatureSelection isteniyor ama 
aynı zamanda biraz ridgenin sağlamlığından da faydalanmak isteniyorsa.


en iyi apha ve l1_ratio değeri için 
from sklearn.linear_model import ElasticNetCV
elastic_cv = ElasticNetCV(cv=5, l1_ratio=[.1, .5, .7, .9, .95, 1])
elastic_cv.fit(X_train, y_train)
print("En iyi alpha:", elastic_cv.alpha_)
print("En iyi l1_ratio:", elastic_cv.l1_ratio_)


'''
from sklearn.linear_model import ElasticNet

# Modeli oluştur
elastic_model = ElasticNet(alpha=1.0, l1_ratio=0.5)

# Eğit
elastic_model.fit(X_train, y_train)

# Tahmin yap
y_pred = elastic_model.predict(X_test)


'''
Logistic Regression:
Sınıflandırma yapmak istediğimiz işlerde kullanabiliriz.
mail spam mı değil mi? Bu hasta kanser mi değil mi?
Çıktı artık sürekli bir sayı değil sınıf olacak.

Linear regression gibi çalışır fakat bu değeri bir de sigmoid fonksiyonu içine verir.
Sigmoid fonksiyonu skoru 0 ile 1 arasında sıkıştırır.

Sigmoid fonksiyonu grafiği s şeklindedir.

İkili sınıflandırma gereken durumlarda kullanılır. 
Çok sınıflı problemler içinde genelleştirilebilir.


ağırlık pozitifse; Özellik değeri arttıkça hadef sınıfa ait olma olasılığı artar,
Ağırlık negatifse; Özellik değeri arttıkça hedef sınıfa ait olma olasılığı azalır

yani bu özelliğin ağırlığı büyük ve pozitifse, o özelliğin büyümesi sınıf 1 olasılığını ciddi artırır.

pozitif ağırlık sınıf 1 olasılığını arttırıyor. Ağırlıkların büyüklükleride etkili
negatif ağırlık sınıf sıfır olma olasılığını arttırıyor.
büyük mutlak değer ise daha güçlü etki sağlıyor.
'''

'''
SGDClassifier (Stochastic Gradient Descent)
Rastgele (stokastik) gradyan inişi

çok büyük veri setleri üzerinde çok hızlı çalışan bir sınıflandırma algoritmasıdır.
Logistic Regression, SVM gibi modelleri SGD kullanarak yaklaşık çözer.

Logistic Regression modeli veri setinin tamamı ile çalışır.
SGDClassifier modeli küçük parçalarla çalışır.

çok hızlıdır.

Neden kullanırız:
Büyük veri setleri varsa. 
RAM yetersizse
hızlı, online veya streaming öğrenmesi gerekiyorsa.

Küçük veri setlerinde logisticRegression daha kararlıdır.

parametreler:
loss: Çözeceğimiz model ile iligi bilgi, (genelde log_loss, hinge) kullanılır.
penalty: regularization tipi(l1,l2,elasticnet)
alpha: regularization kuvveti
max_iter: epoch sayısı.
learning_rate: öğrenme oranı ayarlama 
early_stoping: eğer doğruluk ilerlemiyorsa erken durdur


Kullanırken StandardScaler ile veri ölçekle, learning_reate="adaprive"
seçilirse daha güvenli bir öğrenme. Cross-validation ile en iyi alpha ve learning_rate değerini hesapla
partial_fit() yapısı kullanılabiliyor. buna bak????
'''


'''
SGDRegressor:
Büyük veri setlerinde hızlı çalışa bir modeldir. 
SGD kullanarak linear regression veya diğer regression modellerini
yaklaşık olarak çözer.

Normal regression modelinden farklı olarak veriyi küçük kü.ük eğitime alır.
Amaç sürekli bir değer tahmin etmektir.

parametreler:
loss, penalty, alpha, learning_rate, eta0;başlangıç öğrenme oranı, max_iter, early_stopping
özellikleri standardScaler ile ölçeklendirebiliriz.
learning_rate="adaptive" ile daha güvenli bir öğrenme yapabiliriz.
'''


'''
Perceptron:
en basit yapay sinir ağı modelidir. 
ikili sınıflandırıcıdır.

bir doğru çizer ve veriyi bu doğrunun iki yanına ayırır.
en basit evet hayır veren algoritmadır.

Nasıl çalışır:
1) girdi özelliklerinden bir lineer skor hesaplar. 
sonra bu skoru threshold a uygular. ve bu şekilde sınıflandırma yapar. 

öğrenme süreci:
Başlangıçta ağırlıklar rastgele verilir. 
Eğitim sırasında eğer tahmin yanlışsa, ağırlıklar güncellenir.
doğru tahmin ederse ağırlıklar değişmez, yanlış tahminde ağırlıklar güncellenir.

parametreler: 
max_iter, eta0;başlangıç öğrenme oranı, penaltly, tol;iyileşme durursa erken bitirme

çok basit ve hızlıdır, kolay implementasyon, online learning destekler
sadece lineer ayrılabilir verilerde çalışır, lineer olmayan problemlerde başarısızdır, 
doğrusal olmayan sınırlar çizemiyor. 

Eğer veri lineer ayrılabilir değilse, perceptron başaramaz.

kullanmadan önce veriyi ölçekledir. 
'''


'''
RidgeClassifier:

RidgeClassifier, ridge regression mantığıyla çalışır.

Normal ridge regression sürekli değer tahmin ederdi, ridgeClassifier;
çıktıyı sınıfa dönüştürür.

RidgeClassifier = Linear Regression + Ridge Regularization + Sınıflandırma

Nasıl çalışır: 
model öncelikle doğrusal bir skor üretir, sonra bu skora göre karar verir.
hangi sınıfa yakınsa o sınıfı seçer.

Yani logistic regression gibi olasılık hesaplamaz. Sadece skorları karşılaştırır. 
Sert (hard) karar verir.

RidgeClassifier Neden Vardır?
Özellikle çok yüksek boyutlu veri setlerinde mesela binlerce future L2 
regularization kullanmak overfittingi azaltır. 

LogisticRegression gibi olasılık hesabı yapmak istemiyorsak (sadece doğrudan sınıf tahmini istiyorsak.)
RidgeClassifier çok hızlı çalışır.

parametreler:
alpha, fit_intercept;bias terimi ekle, class_weight;sınıflar arasında ağırlıkladırma yap. dengesiz sınıflar için 
kullnışlı.
solver;çözüm yöntemi
Çok boyutlu veya düzenli bir sınıflandırıcı lazımsa ridgeClassifier uygundur.
Kullanırken ölçeklendirme yap. StandardScaler veya MaxMinScaler kullanılabilir.
'''


'''
PassiveAggressiveClassifier:
Çok büyük, akan veri için tasarlanmış online learning algoritmadır.

Eğer tahmin doğruysa pasif kalır. 
Eğer tahmin yanlışsa agresif bir güncelleme yapar. 
Yanlış tahminde ağırlıklar güçlü bir şekilde değiştirilir.

çok sık değişen veriler için uygundur.

Linear bir model kurar, eğer tahmin doğruysa hiçbir şey yapmaz , eğer tahmin yanlışsa hızlı ve agresif bir şekilde ağırlıkları günceller. 

parametreler:
C = regularization kuvveti
loss= hinge veya squared_hinge
max_iter
early_stopping
shuffle = eğitim sırasında örnekler karıştırılsın mı 
gürültülü veride kararsız sonuç verebilir.
veriyi normalize et. 
c değerini gridSearch ile ayarlayabiliriz.
partial_fit ile gerçek zamanlı öğrenme yapılabilir. 
'''