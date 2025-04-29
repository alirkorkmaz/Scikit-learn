from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree
)


'''
DecisionTreeClassifier:
Ağaç yapısına benzeyen bir modeldir.
Model veriyi özelliklere bölerek sınıflandırma yapar.
if else kuralları gibi çalışır.

karar ağcı veriyi adım adım bölerek hangi sınıfa ait olduğunu belirler. 

Başta tüm veri havuzdadır. model en iyi bölmeyi bulur.
hangi özellikte ve hangi eşikte veri daha iyi ayrılıyor.
bunu ölçmek için impurity ölçer.
gini impurity (varsayılan)
entropy (bilgi kazancı)

veriyi böler. 
alt gruplarda tekrar aynı işlem yapılır.
yaprak düğümler leaf nodes tamamen saf olana kadar devam eder. 

her veri noktasının hangi yaprakta olduğu bulunur.

DecisionTreeClassifier kullanım alanları:
Kategorik sınıflandırma ,tıbbi teşhis sistemi
kredi risk analizi,
özellik seçimi

parametreler:
criterion; bölme kalitesini ölçmek için kullanılan yöntem(gini, entropy)
max_depth; ağacın max derinliği
min_samples_split; bir düğümü bölmek için gereken min örnek sayısı
min_samples_leaf;bir yaprakta bulunması gereken minimum örnek sayısı
max_features;bölme sırasında kullaınlacak max özellik sayısı
random_state; sonuçların tekrarlanabilir olması 

yorumlaması kolay, özelliklerin önemini verir, kategorik ve sayısal verilerle çalışır, veri ölçeklendirme gerekmez

aşırı öğrenmeye çok açıktır, küçük değişikliklerde çok farklı ağaç oluşur,
çok derin ağaçlar gereksiz karmaşık olabilir.



'''


'''
DecisionTreeRegressor:

veriyi özelliklere bölerek sürekli bir değer tahmin eder.

yapısı classifier ile aynıdır.

herhangi bir sınıf seçmiyoruz.
ağacı kurar, her yaprakta veri örneklerinin ortalamasını alır.
tahmin değeri olarak bu ortalamayı verir.
sınıf tahmini yapmak yerine sayı tahmini yapar. 



'''


'''
plot_tree()

modellerini grafiksel olarak çizmeye yarar. 
modelin nasıl bölme yaptığı, hangi koşullarla karar verdiği yapraklarda ne değer olduğu
gibi bilgileri görsel bir ağaç yapısında gösterir.

her node bir düğüm noktasıdır.

plot_tree : ağacı adım adım gözle görünür hale getirir.
'''