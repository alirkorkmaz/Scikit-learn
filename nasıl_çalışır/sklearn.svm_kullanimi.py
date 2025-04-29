from sklearn.svm import (
    SVC,
    SVR,
    LinearSVC,
    LinearSVR
)

'''
SVC: 

Support Vector Classifier:
sınıflandırma için kullanılan bir versiyondur.
veirleri bir hiper düzlem ile ayırır

iki sınıfı en iyi ayıran maksimum marj oluşturmaya çalışır.
Sadece sınırdaki örnekler modelin kararını belirler. 

parametreler:
kernel; hangi tür kernel kullanılacak, (linear, poly, rbf, sigmoid)
C; regularization parametresi
gamma; poly kernel için etki alanı
degree; polinomal kernel için polinom derecesi
probability: olasılık tahmini istiyorsak True olmalı
class_weight: sınıf ağırlıklarını ayarlamak için 

linear: veriler doğrusal ayrılabiliyorlarsa: 
rbf: veriler doğrusal değilse,
poly: veriler polinom bir sınırla ayrılıyorsa 
sigmoid: sinir ağı benzeri davranış

karmaşık veri kümelerini çok iyi ayırır
Doğrusal olmayan verilerde kernel tric ile başarılıdır.
overfittige karşı dayanıklıdır.

Kullanırken özellikleri ölçeklemek önemli.(standardscaler)


'''


'''
SVR: Support Vector Regressor
SVM algoritmasının regression için uyarlanmış hali.

veriyi bir tüp içine almaya çalışır. tüpün dışındaki veriler için ceza uygular. 

karmaşık regression problemlerinde güçlüdür. kernel trick sayesinde esnektir.
overfittinge karşı dayanıklıdır. 
büyük veri setlerinde yavaş çalışır.


'''



'''
LinearSVC:
büyük veri setlerinde SVC yerine tercih edilir. svc 'nin doğrusal versiyonudur. 
Yani verileri doğrusal bir sınırla ayırır. 
SVC gibi çalışır ama farklı bir çözüm algoritması kullandığı için çok daha hızlıdır. 

'''

'''
LinearSVR:

SVR algoritmasının sadece doğrusal çalışan versiyondur. SVR gibi çalışır fakat daha hızlıdır.
Sürekli değer tahmini yapar, ancak doğrusal ilişkileri yakalamak için tasarlanmıştır.

'''