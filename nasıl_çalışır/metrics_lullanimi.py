from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    precision_recall_curve,
    make_scorer
)


'''
Sınıflandırma modelinin doğru tahmin oranını hesaplar. 
Kaç tane örneği doğru sınıflandırmış olduğunu gösterdi.

accuracy = Doğru Tahmin / Toplam Örnek Sayısı

y_ture: Gerçek sınıf etiketleri 
y_pred: Tahmin edilen sınıf etiketleri
normalize: True ise oran döner, False ise doğru tahmin sayısı döner. 
sample_weight: Her örneğe ağırlık verilebilir. 

Ne Zaman Kullanılır?
Sınıflandırma problemlerinde ilk değerlendirme ölçütü olarak
genel başarıyı görmek istediğinden kullanılır. 
Veri seti dengeli (balanced) ise accuracy_score anlamlıdır. 

Dikkat edilmesi gerekenler: 
Sınıflar dengesiz (imbalanced) ise yanıltıcı olabilir. 
Örneğin %95 i "0" olan bir veri setinde hep "0" tahmin eden bir model 
%95 doğruluk alır ama bu iyi bir model değildir. 
Bu durumlarda precision, recall, f1_score gibi anlamlı 
metriklerede bakmak gerekir. 
'''
from sklearn.metrics import accuracy_score
y_true = [0,1,1,0,1]
y_pred = [0,1,0,0,1]

accuracy = accuracy_score(y_true, y_pred)
print("accuracy:", accuracy)


'''
Precision_score:
modelin pozitif tahminlerinin ne kadarının gerçek pozitif olduğunu gösterir. 
Model, pozitif dediğinde ne kadar isabetliydi?

Precision = True Pozitif / True Pozitif + False Pozitif

TP = Gerçekten pozitif olan ve doğru tahmin edilen 
FP = Gerçekte negatif ama pozitif olarak tahmin edilmiş

y_true: gerçek etiketler
y_pred: Tahmin edilen etiketler
labels: Hangi etiketler için score değeri hesaplanacak
pos_label: pozitif sınıf etiketi (varsayılan 1)
average: çok sınıflı durumalr için binary, macro, micro, weighted
zero_division: 0'a bölme durumunda ne dönecek

Ne zaman kullanılır: 
Yanlış pozitiflerin önemli olduğu durumlarda, 
Spam e-posta tespiti: bu spam dediğimizde gerçekten spam mı 
Hastalık taraması: Hasta dediğimiz kişilere gereksiz panik veya 
test yapılmasını istemeyiz.
Dolandırıcılık tespiti: Masum işlemleri yanlışlıkla şüpheli dememek için 

Dikkat edilmesi gereken noktalar: 
Precision, yüksek olabilir ama recall çok düşükse, bu model çok az tahmin yapıyor olabilir. 
Bu yüzden recall_score ve f1_score ile birlikte bakılır.
'''

from sklearn.metrics import precision_score

y_true = [0,1,1,0,1,0]
y_pred = [0,1,0,0,1,1]

precision = precision_score(y_true, y_pred)
print(precision)


'''
Recall_score:
modelin gerçek pozitifleri ne kadar yakalayabildiğini gösterir.

racall = True Pozitif / True Pozitif + False Negatif

False Negatif: Gerçekten pozitif ama modelin kaçırıdığı 

Parametreler: 
y_true: 
y_pred:
average: 'binary', 'macro', 'micro', 'weighted', 'sample'
pos_label: Pozitif sınıf etiketi
zero_division:

Ne zaman kullanılır?

Kaçırılan pozitiflerin maliyeti yüksekse racall önemlidir.
Kanser taraması: Hasta olanları atlamamak gerek.
Sahtekarlık tespiti: Gerçek dolandırıcı işlemleri kaçırmamak gerek
Yangın algılama: Gerçek yangını gözden kaçırmamalıyız.

Dikkat edilmesi gerekenler: 
Recall yüksek olabilir ama precision düşükse, model çok fazla pozitif diyor olabilir. 
Bu da gereksiz alarm anlamına geliyor. 
O yüzden precision ve recall genelde birlikte değerlendirilir. 
Bu ikisinin dengesi için genelde bir sonraki adım olarak f1_score kullanılır
'''

from sklearn.metrics import recall_score

y_true = [0, 1, 1, 1, 0, 1]
y_pred = [0, 1, 0, 1, 0, 0]

recall = recall_score(y_true, y_pred)
print("Recall:", recall)


'''
f1-score: 

precision ve racall skorlarının harmonik ortalamsını alır. 
Dengeli bir şekilde hem doğru pozitifleri yakalamak, hem de yanlış pozitiflerden kaçınmak isteyen modellerde kullanılır. 

f1-score = 2 ((precision . Recall) / (Precision + Recall))

Ne zaman kullanılır: 
Hem false pozitif hem de false negatif öenmliyse.
Veri dengesizse 

Dikkat Edilmesi Gerekenler:
F1 score yüksekse, bu model hem pozitifleri iyi yakalıyor hem de çok fazla yanlış pozitif yapmıyor demek. 

Ama bazı uygulamalarda F1 skoru yerine sadec recall yada precision daha önemli olabilir. 

Eğer sınıflar dengesizse, avarage="macro"  veya everage="weighted" gibi ayarlar kullanılmalıdır. 

'''

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, f1_score {f1:.2f}")


'''
Confusion Matrix:

Ne işe yarar?
Bir modelin sınıflandırma performansını detaylı görmek için kullanılır.
Tahminlerin doğruluğunu 4 kategoriye ayırırız.
TP: Gerçek pozitifleri doğru tahmin.
TN: Gerçek negatifleri doğru tahmin.
FP: Yanlışlıkla pozitif tahmin edilen negatifler.
FN: Yanlışlıkla negatif tahmin edilen pozitifler

y_ture:
y_pred:
labels: Sınıfların sıralaması
normalize: None, True, pred, all yüzdelik normalize eder. 

Ne zaman kullanılır? 
Persormansı daha detaylı analiz etmek istediğimizde. 
Sadece accuracy değil, hataların nerelerde yapıldığını görmek istiyorsak. 
Özellikle dengesiz sınıflarda çok faaydalıdır.

Dikkat edilmesi gerekenler:
confusion_matrix, tek başına bir başarı ölçütü değildir ama diğer metriklerin 
temelidir.
FP ve FN sayısı hataların tipini gösterir.
Özellikle medikal, güvenlik veya kredi sistemlerinde, FP ve FN maliyeti farklı olabilir.

Çıktı örneği:
[[2 1]
 [1 3]]

bu çıktının anlamı:
           Tahmin
          0    1
Gerçek  0 [TN  FP]
        1 [FN  TP]

'''
from sklearn.metrics import confusion_matrix

y_true = [1, 0, 1, 1, 0, 0, 1]
y_pred = [1, 0, 0, 1, 0, 1, 1]

cm = confusion_matrix(y_true, y_pred)
print(cm)




'''
classification_report: 

bir sınıflandırma modelinin genel performansını tek bir tabloda özetler. 
precision, recall, f1-score, support(her sınıftan kaç tane doğrı var.)

Her sınıf için ayrı ayrı ve toplam ortalamalarla birlikte rapor verir.

parametreler:
y_ture:
y_pred:
labels: Raporlanacak sınıf etiketleri
target_names: Sınıfların isimleini (["kedi", "köpek"])
output_dict=True ile sözlük formatında  çıktı almak için
zero_devision:sıfıra bölme hatalarında ne yapılacağı

Ne zaman kullanılır?
Birden fazla sınıf varsa mükemmel araçtır. 
Precision, recall, f1 skorunu tek tabloda görmek istiyorsan ideal.
Dengesiz sınıflarda performansı detaylı kıyaslamak için.

Dikkat Edilecek Noktalar: 
support, o sınıftan kaç örnek var. Az destekli sınıflarda metrikler yanıltıcı olabilir. 
macro avg: her sınıfın eşit öneme sahip olduğunu varsayar.
weighted_avg: sınfıların gözelem sayısnı göre ağırlıklı ortalamasıdır.
çok sayıda sınıf varsa, output_dict=True ile tabloyu pandas.DataFrame çevirmek analiz için çok kullanışlıdır. 

örnek çıktı:

              precision    recall  f1-score   support

           0       1.00      1.00      1.00         1
           1       0.50      0.50      0.50         2
           2       1.00      0.67      0.80         3

    accuracy                           0.75         6
   macro avg       0.83      0.72      0.77         6
weighted avg       0.83      0.75      0.77         6


'''

from sklearn.metrics import classification_report

y_true = [0, 1, 2, 2, 2, 1]
y_pred = [0, 0, 2, 2, 1, 1]

print(classification_report(y_true, y_pred))

'''
roc_auc_score:

roc_auc_score, Roc eğirisi altında kalan alanı (AUC = Area Under the Curve)
hesaplar. Bu skor bir modelin pozitif sınıfları diğerlerinden ne kadar iyi ayırt ettiğini ölçer. 
ROC eğrisi: 
Gerçek pozitif oranı ile yanlış pozitif oranı arasındaki ilişkiyi hesaplar. 

ROC AUC skoru 1 ise mükemmel model
ROC AUC skoru 0.5 ise rastgele tahmin gibi
ROC AUC skoru < 0.5 ise hatalı bir model 

roc_auc_score etiketleri değil, tahmin edilen olasılıkları ister.
Yani predict çıktısı ile değil predict_proba()
veya decision_function çıktısı kullanılır. 

parametreler: 
y_true:
y_score:
average: çoklu sınıflar için macro, weightd 
multi_class: ovr (one vs rest) veya ovo (one vs one )
max_fpr: isteğe bağlı, eğriyi belli bir yanlış pozitif oranına kadar sınırla

Ne zaman kullanılır: 
binary sınıflandırmada performans ölçmek için kullanılır.
Sınıf dengesizliği varsa Accuracy kullanmak yanıltıcı olur, bu yüzden ROC AUC çok daha doğru sonuç verir. 
iyi bir model ROC eğrisi altında büyük bir alan bırakır.

Dikkat edilecek noktalar: 
Sadece .predict() ile etiket verip roc_auc_score çalıştıramk hata verir. Veya yanlış sonuç üretir. 
.predict_proba() çıktısından pozitif sınıf ([:, 1]) seçilmelidir.

ROC AUC = Kümeler iyi ayrışıyor mu sorusunun matematiksel cevabıdır.
Çok sınıflı problemlerde multi_class = "ovr" parametresi verilmelidir. 

Özetle: 
roc_auc_score, özellikle dengesiz veri üzerinde modeli değerlendirirken 
güvenilir bir metriktir. Accuracy yüksek olsa bile ROC AUC düşükse model kötü olabilir. 

'''


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = [[0.1], [0.4], [0.35], [0.8]]
y = [0, 0, 1, 1]

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42)

model= LogisticRegression()
model.fit(x_train, y_train)
y_pred_prob = model.predict_proba(x_test)[:, 1]
print(roc_auc_score(y_test, y_pred_prob))





'''
roc_curve:

roc_curve; roc eğrisini çizmek için gerkeen TPR ve FPR değerlerini verir. 
Bu değerlerle ROC eğrisini matplotlib ile kolayca çizebiliriz.

ROC eğrisi, sınıflandırma eşiklerinin değişmesiyle TPR ve FPR'in nasıl değiştiğini gösterir.

parametreler: 
y_true:
y_score: pozitif sınıfın olasılık tahminleri predict_proba[:, 1]
pos_label: Pozitif sınıfı belirle 
drop_intermediate: Gereksiz eşikleir atlar. 

fpr: yanlış pozitif oranları 
tpr: doğru pozitif oranları recall 
thresholds: Her noktaya karşılık gelen karar eşikleri. 

Nerede kullanılır:
Roc eğrisi çizmek için, 
ROC AUC skoru ile birlikte kullanmak yaygındır. 
Modelin eşik değerini değiştirerek farklı sonuçlarını gözlemlemek için
Sınıf dengesizliği olan veri setlerinde ideal eşik değeri belirlemek için. 

Etiket değil olasılık değerleri verirlir -> .predict_proba()[:,1]
ROC eğrisi, modelin tahmin eşiğine göre nasıl değiştiğini görmeyi sağlar. 
thresholds dizisi kullanılarak istenilen eşik değeri manuel olarak seçilebilir.

ROC eğrisi nerede kullanılır. 
Modeli kıyaslamak için ROC üst üste çizilir. 
Eşik değer seçimi gerektiğinde ROC eğirisi rehber olur.

Kısaca roc_curve, bir modelin sınıflandırma eşiklerine göre nasıl davrandığını göstermek için mükemmel bir araçtır. 

'''

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Veri oluştur
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Olasılık tahminleri
y_proba = model.predict_proba(X_test)[:, 1]

# ROC Eğrisi
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# ROC AUC skoru
auc = roc_auc_score(y_test, y_proba)

# Çizim
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], '--', color='gray')  # random çizgisi
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.grid(True)
plt.show()


'''
mean_squared_error:

ne işe yarar?

mean_squared_error, tahmin edilen değerlerle gerçek değerler arasındaki
farkların karesinin ortalamasını verir. 

Bu metrik modelin tahmin hatalarınının büyüklüğünü ölçmek için kullanılır.

Farkların karesi alındığı için büyük hatalar daha fazla cezalandırılır.

parametreler: 
y_true:
y_pred:
sample_weight: her örnek için ağırlık
squared: True, eğer false yapılırsa karakök ortalama hata döner

hangi durumda kullanılır?

Regresyon modellerinde modelin genel hatasını ölçmek için kullanılır, 
Eğer büyük hatalar daha önemliyse, MSE kullanılır(çünkü kare alındığı için büyük hatalar daha çok cezalandırılır.)

Modeli karşılaştırmak için kullanılır. 
'''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# Veri oluştur
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)

# Eğitim/test ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modeli eğit
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# MSE hesapla
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.2f}")


'''
mean_absolute_error:

tahmin edilen ile gerçek değerler arasındaki mutlak farkların ortalamasını verir. 
Tahminlerimiz ne kadar sapmış? Ortalama ne kadar hata yapıyoruz? bu sorualara yanıt verir.

Her hatayı kareye alamak yerine mutlak değerini alır. 
Böylece aykırı değerlere daha az duyarlı olur.


'''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error

# Veri seti
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=0)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model eğitimi
model = LinearRegression()
model.fit(X_train, y_train)

# Tahminler
y_pred = model.predict(X_test)

# MAE hesapla
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")


'''
r2_score:

ne işe yarar? 
modelin hedef değişkendeki değişimi ne ölçüde açıkladığını (yani verinin ne kadarını
açıkladığını ) ölçer. 

r2_score bize, modelin tahminlerinin ortalama tahminlerden ne kadar daha iyi olduğunu söyler. 

Hangi durumlarda kullanılır?

regresyon problemlerinde başarıyı genel olarak ölçmek için 
özellikle değişkenliğin ne kadarını açıkladığını görmek istediğimizde,
diğer regresyon metrikleriyle birlikte yorumlandığunda anlam kazanır.

r2_score ve overfitting: 
Eğitim verisi üzerinde çok yüksek r^2 ama testte düşükse, overfitting olabilir.
Model karmaşıklığı arttıkça r^2 yükselir gibi görünsede genelleme yeteneği düşebilir. 

r2_score ile dikkat edilmesi gerekenler: 
Negatif çıkabilir; bu modelin ortalamadan bile kötü yaptığı anlamına gelir.
r^2 her zaman güvenilir bir kalite göstergesi değildir. mutlaka MSE MAE gibi metriklerle desteklenmelidir. 
Non-linear modellerde düşük r^2 'ler yanıltıcı olabilir. 


'''

from sklearn.metrics import r2_score

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

r2 = r2_score(y_true, y_pred)
print(f"R2 Score: {r2:.2f}")

'''
precision_recall_curve:
Sınıflar dengesiz olduğunda çok işimize yarayan bir metriktir.
farklı eşik (threshold) değerine göre precision ve recall değerlerini hesaplar.

Özellikle pozitif sınıfın nadir(Rare class) olduğu dengesiz veri setlerinde ROC eğrisi yerine genellikle bu eğri tercih edilir. 
Bir sınıflandırıcı, pozitif sınıf için bir olasılık tahmini (perdict_proba) verir.

Eşik ↑ → Daha seçici model → Precision ↑, Recall ↓
Eşik ↓ → Daha toleranslı model → Recall ↑, Precision ↓

Sınıf dengesizliği varsa, pozitif sınıf kaçrımak istemiyorsak
threshold seçimi yapmak istiyorsak. 

Precision-Recall eğrisinin yorumu:
Kıvrımı daha yukarda ve sağda olan bir eğri daha iyi bir model demektir. 
Eğrinin altında kalan alan büyükse PR-AUC -> model daha  iyi

ROC-AUC yanıltıcı olabilir. PR eğrisi nadiren yanıltır dengesiz veri setlerinde. 
y_score olarak predict_proba kullanılır
'''

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Gerçek etiketler ve tahmin olasılıkları (pozitif sınıf için)
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Görselleştirme
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()


'''
make_scorer:

kendi yazdığımı bir değerlendirme fonksiyonunu, GridSearchCV, Cross_val_score, cross_validate gibi
yerlerde scoring olarak kullanılabilir hale getirmek.

kendi yazdığımız min(precision, recall) gibi özel bir metrik varsa
Varsayılan metrikeler dışında değerlendirme yapmak istirsak,
sadece t_true, y_pred alan fonksiyonları GridSearch'te kullanmak istiyorsak

make_scorer(score_func, *, greater_is_better=True, needs_proba=False, needs_threshold=False, **kwargs)
parametreler:
score_func: kendi yazdığımız veya kullandığımız skor fonksiyonu 
greater_is_better: True ise skor yüksek olmalı, False ise daha düşük değer daha iyidir.
needs_proba: Fonksiyonun predict_proba ile mi çalıştığını belirtir. 
needs_threshold: decision_function çıktısı kullanılacaksa True olur
**kwargs: skor fonksiyonuna ekstra parametre göndermek istersek.

GridSearchCV, cross_val_score, cross_validate işlemlerinde kullanılır.

make_score ile yazılan fonksiyon her fold için çağrılır. bu yüzden hızlı olması gerekir.
Needs_proba=True; ayarlandığında predict_proba olmayan modellerde hata alırız.
ÖZellile özelleştirilmiş f1, penalize edilmiş loss, domain-specific puanlama sistemleri gibi
durumlar için çok gereklidir.
'''

from sklearn.metrics import precision_score, recall_score

def min_precision_recall(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return min(p, r)

custom_scorer = make_scorer(min_precision_recall)