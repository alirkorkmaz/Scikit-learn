from sklearn.utils import  (shuffle,
                            resample,
                            bunch,
                            check_random_state,
                            check_X_y,
                            check_array,
                            Memory,

                            )


'''
Shuffle: Verileri rastgele karıştırır. 


resample: Veriden bootstrap örneklemesi yapar

check_X_y: veri girişlerini kontrol eder, x ve y uygun mu

compute_class_weight: sınıflar arasında denge sağlamak için sınıf ağırlıklarını hepsaplar. 
özellikle dengesiz sınfılar için çok faydalıdır.

memory: hafızada önbellekleme yapar. caching
Ağır işlemleri hızlandırır.
Örneğin pipeline içindeki hesaplamaları kaydetmek için kullanılır. 


'''