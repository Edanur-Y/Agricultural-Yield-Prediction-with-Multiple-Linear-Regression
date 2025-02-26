Çoklu Doğrusal Regresyon ile Tarımsal Verim Analizi
================
Edanur Yılmaz,
2023-11-26

## İçerik
- [Veri İnceleme](#veri-i̇nceleme)
- [Kayıp Gözlemler](#kayıp-gözlemler)
- [Model Oluşturma](#model-oluşturma)
- [Aykırı Değer Kontrolü](#aykırı-değer-kontrolü)
- [Model Karşılaştırması](#model-karşılaştırması)
- [Test Seti üzerinden model
  değerlendirme](#test-seti-üzerinden-model-değerlendirme)
- [Çoklu Bağlantı Sorunu
  (Multicollinearity)](#çoklu-bağlantı-sorunu-multicollinearity)
- [İlişkili Hatalar](#i̇lişkili-hatalar)

Gerekli paketler aktif hale getirildi.

``` r
library(ggplot2)
library(dplyr)
library(broom)
library(ggpubr)
library(ISLR)
library(PerformanceAnalytics)
library(ggmice)
library(mice)
library(caret)
library(car)
library(lmtest)
```

## Veri İnceleme
<sup>[İçeriğe dön.](#İçerik)</sup>

Kaynak veri seti: [Synthetic Agricultural Yield Prediction
Dataset](https://www.kaggle.com/datasets/blueloki/synthetic-agricultural-yield-prediction-dataset)

Veri seti hiç eksik veri bulundurmuyordu. Veri içinden rastgele
değerlere NA atadım. Bu işlemin ne kadar doğru olduğunu bilmemekle
beraber veri seti büyük olduğu için herhangi bir sorun teşkil ettiğini
düşünmüyorum. Kullandığım yöntemde her sütuna atanacak NA miktarını
belirleyemediğim için bağımlı değişken olan Yield_kg_per_hectare
verisini bu işleme sokmadım.

Düzenlenmiş veri
seti: [agricultural_yield](https://drive.google.com/file/d/1HsFb9ZiuRqXHNGtZFF6sCf1AddO2-dTc/view?usp=drive_link)

Tarımsal verim tahmini için hazırlanmış sentetik bir veri seti
inceliyoruz. Veri setinin, toprak kalitesi(50-100), tohum
çeşitliliği(0-1), hektar başına kullanılan gübre(kg), güneşli gün
sayısı, yağış(mm), sulanma sayısı, hektar başına düşen tarımsal
verimlilik(kg) değişkenlerine sahip olduğunu görüyoruz.

Değişkenlerden ***Seed_Variety***’nin binary olmasından ve
***Irrigation_Schedule***’ın kesikli olmasından dolayı bu iki değişkeni
veri setinden çıkarıyorum.

``` r
#View(agricultural_yield)
agricultural_yield<-subset(agricultural_yield, select = -c(Seed_Variety, Irrigation_Schedule))
names(agricultural_yield)
```

    ## [1] "Yield_kg_per_hectare"             "Fertilizer_Amount_kg_per_hectare"
    ## [3] "Sunny_Days"                       "Rainfall_mm"                     
    ## [5] "Soil_Quality"

Tahmin edilmek istenen değişken: ***Yield_kg_per_hectare***  
Bağımsız değişkenler: ***Soil_Quality, Rainfall_mm,
Fertilizer_Amount_kg_per_hectare, Sunny_Days***

``` r
cor(na.omit(agricultural_yield))
```

    ##                                  Yield_kg_per_hectare
    ## Yield_kg_per_hectare                       1.00000000
    ## Fertilizer_Amount_kg_per_hectare           0.28362403
    ## Sunny_Days                                 0.09463974
    ## Rainfall_mm                               -0.24695586
    ## Soil_Quality                               0.10622836
    ##                                  Fertilizer_Amount_kg_per_hectare   Sunny_Days
    ## Yield_kg_per_hectare                                  0.283624032  0.094639738
    ## Fertilizer_Amount_kg_per_hectare                      1.000000000  0.007313751
    ## Sunny_Days                                            0.007313751  1.000000000
    ## Rainfall_mm                                           0.003010412 -0.004077846
    ## Soil_Quality                                         -0.005276133 -0.004986840
    ##                                   Rainfall_mm Soil_Quality
    ## Yield_kg_per_hectare             -0.246955865  0.106228361
    ## Fertilizer_Amount_kg_per_hectare  0.003010412 -0.005276133
    ## Sunny_Days                       -0.004077846 -0.004986840
    ## Rainfall_mm                       1.000000000  0.011085051
    ## Soil_Quality                      0.011085051  1.000000000

Korelasyon matrisi incelendiğinde bağımlı
değişkenin(Yield_kg_per_hectare) Rainfall_mm ile negatif diğer bağımsız
değişkenler ile ise pozitif ilişkisi olduğu görülüyor.  
Bağımsız değişkenlerin kendi aralarında güçlü ilişkileri olmadığı
görülmektedir. Çoklu doğrusal bağlantı(multicollinearity) riski
taşımıyor.  
Bağımsız değişkenlerin bağımlı değişkenle aralarında da çok güçlü bir
ilişki olmadığını görebiliyoruz.

``` r
pairs(na.omit(agricultural_yield), pch=19)
```

![](Çoklu-Doğrusal-Regresyon-ile-Tarımsal-Verim-Analizi_files/figure-gfm/visual_correlation-1.png)<!-- -->

``` r
chart.Correlation(na.omit(agricultural_yield),histogram = TRUE, pch=19)
```

![](Çoklu-Doğrusal-Regresyon-ile-Tarımsal-Verim-Analizi_files/figure-gfm/visual_correlation-2.png)<!-- -->

Serpilme diyagramlarına(scatter plot) bakıldığında bağımsız değişkenler
arasındaki pozitif veya negatif ilişkilerin açık olmadığı görülmektedir.
Gözlem sayısının fazla olmasından da kaynaklı olarak veri noktaları çok
saçılmış gözükmemekte yani görsel olarak aralarındaki ilişkinin çok
düşük olduğunu söylemek zor. Bağımsız değişkenlerin bağımlı değişkenle
aralarındaki ilişkinin anlamlı olduğunu(\*\*\*) görebiliyoruz.

## Kayıp Gözlemler
<sup>[İçeriğe dön.](#İçerik)</sup>

``` r
plot_pattern(agricultural_yield, square=FALSE, rotate=TRUE)
```

![](Çoklu-Doğrusal-Regresyon-ile-Tarımsal-Verim-Analizi_files/figure-gfm/na_pattern-1.png)<!-- -->

18639 gözlemde NA değeri bulunmamaktadır. 393 gözlem için sadece
Soil_Quality değişkeninde, 206 gözlem için sadece Rainfall_mm
değişkeninde, 153 gözlem için sadece Sunny_Days değişkeninde ve 104
gözlem için sadece Fertilizer_Amount_kg_per_hectare değişkeninde NA
değerleri olduğu görülmektedir. Veri setinde toplamda 1971 NA değeri
bulunmaktadır.

Bağımlı değişkende eksik gözlem bulunmaması ve ne veri seti genelinde ne
de değişkenler içerisinde bulunan eksik gözlem sayısının çok büyük
olmaması nedeniyle eksik verilerin doldurulabilir olduğunu
görebiliyoruz. mice fonksiyonunu kullanarak eksik değerleri
dolduruyoruz.

``` r
imputed<-mice(agricultural_yield)
```

``` r
agricultural_yield_Imp<-complete(imputed,4)
sum(is.na(agricultural_yield_Imp))
```

    ## [1] 0

Bu işlemden sonra NA değeri kalmadığını görebiliyoruz.

## Model Oluşturma
<sup>[İçeriğe dön.](#İçerik)</sup>

Modelimizin iyi çalışıp çalışmadığını deneyebilmek için eğitim ve test
setlerine ihtiyacımız var. Eğitim ve test setleri oluşturmak için veri
setini iki parçaya bölüyoruz.

``` r
set.seed(44)
sampleIndex<-sample(1:nrow(agricultural_yield_Imp),size=0.8*nrow(agricultural_yield_Imp))
trainset<-agricultural_yield_Imp[sampleIndex,]
testset<-agricultural_yield_Imp[-sampleIndex,]
```

Modeli oluşturalım.

``` r
model1<-lm(Yield_kg_per_hectare ~
             Soil_Quality+Rainfall_mm+Fertilizer_Amount_kg_per_hectare+
             Sunny_Days, data=trainset)
summary(model1)
```

    ## 
    ## Call:
    ## lm(formula = Yield_kg_per_hectare ~ Soil_Quality + Rainfall_mm + 
    ##     Fertilizer_Amount_kg_per_hectare + Sunny_Days, data = trainset)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -637.07 -125.05   22.81  129.79  669.54 
    ## 
    ## Coefficients:
    ##                                   Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                      523.19497   18.37042   28.48   <2e-16 ***
    ## Soil_Quality                       1.61231    0.10063   16.02   <2e-16 ***
    ## Rainfall_mm                       -0.49375    0.01457  -33.89   <2e-16 ***
    ## Fertilizer_Amount_kg_per_hectare   0.78981    0.02027   38.97   <2e-16 ***
    ## Sunny_Days                         1.78963    0.14631   12.23   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 184.5 on 15995 degrees of freedom
    ## Multiple R-squared:   0.16,  Adjusted R-squared:  0.1598 
    ## F-statistic: 761.5 on 4 and 15995 DF,  p-value: < 2.2e-16

Model sonuçları incelendiğinde $R^2$ $\approx$ 0.16 olarak elde
edilmiştir. $R^2$ değeri küçüktür.(Veriyi seçtikten sonra bu
aşamada anlamlılığı yeterli buldum sanırım fakat $R^2$ değeri
modelin iyi tahminlerde bulunamayabileceğini gösteriyor.) Bunun yanısıra
model anlamlı çıkmıştır *(p\<2.2e−16)*. Tüm değişkenlerin anlamlı
olduğu görülmektedir(\*\*\*).

Modeli grafikler üzerinden inceleyelim.
![](Çoklu-Doğrusal-Regresyon-ile-Tarımsal-Verim-Analizi_files/figure-gfm/plot_model1-1.png)<!-- -->![](Çoklu-Doğrusal-Regresyon-ile-Tarımsal-Verim-Analizi_files/figure-gfm/plot_model1-2.png)<!-- -->

**Residuals vs Fitted:** x ekseninde fitted($\hat{y}$) ve y
ekseninde residuals($y - \hat{y}$) değerleri olan bu grafik
modelin doğrusallığını ve varyansın homojenliğini(homoscedasticity)
tespit etmekte kullanılır. model1’in Residuals vs Fitted grafiğini
incelediğimizde kırmızı model çizgisinin 0 üzerinde bozulmadan gittiğini
ve artıkların dağılımının x ekseni boyunca yaklaşık olarak aynı olduğunu
görebiliyoruz. Model doğrusal ve varyansı homojen gözüküyor.

**Q-Q Residuals:** Modelin doğrusal regresyon modeline uyup uymadığını
kontrol etmekte kullanılır. Veri noktaları yaklaşık olarak doğrusal
regresyon çizgisinin üstünde ise modele uyar.  
model1’in Q-Q Residuals grafiğini incelediğimizde veri noktalarının
yaklaşık olarak doğrusal regresyon çizgisinin üstünde olduğunu
görebiliyoruz. Modelin özellikle sağ üst ve sol alta doğru bozulduğunu
ama genel anlamda doğrusal regresyon modeline uyduğunu söyleyebiliriz.

**Scale-Location:** x ekseninde fitted ve y ekseninde
standartlaştırılmış artık değerlerin karekökü olan bu grafik, residuals
vs fitted grafiğine benzerdir. Varyansın homojenliği(homoscedasticity)
analizini basitleştirir.  
model1’in bu grafiğe residuals vs fitted grafiğine benzer şekilde
uyduğunu görebiliyoruz.

**Residuals vs Leverage:** Doğrusal regresyon için
fitted($\hat{y}_i$) değerin gerçek($y_i$) değerdeki
değişime olan hassasiyetini(leverage) ölçmede kullanılır. Model,
leverage arttıkça standartlaştırılmış artık değerlerin dağılımının nasıl
değiştiğini gösterir. Heteroskedasticity ve nonlineer’liğin tespit
edilmesinde de kullanılabilir. Yüksek leverage’a sahip noktalar model
üzerinde etkili olabilir, silindiğinde modelin çok değişimesine neden
olabilir. Bu noktaların tespit edilmesinde de Cook’s distance
kullanılır. Kırmzı çizginin dışında kalan noktalar model üzerinde yüksek
etkiye sahiptir. model1’in grafiğine bakıldığında Cook’s distance
çizgilerini göremiyoruz.

``` r
max(cooks.distance(model1))
```

    ## [1] 0.001481558

Maksimum Cook’s distance değerine baktığımzda ise 0.5’in çok altında bir
değerde olduğunu görüyoruz. Model Cook’s distance çizgilerinin
içindedir.

## Aykırı Değer Kontrolü
<sup>[İçeriğe dön.](#İçerik)</sup>

Aykırı değerleri bulmak için Cook’s distance kullanalım.
Uzaklığı(distance) bulmak için iki ölçüt vardır.

``` r
dist<-cooks.distance(model1)
olcut1<- mean(dist)*3
olcut2<-4/length(dist)
```

``` r
olcut1
```

    ## [1] 0.0001883991

``` r
olcut2
```

    ## [1] 0.00025

İki ölçüt arasında çok büyük bir fark yok fakat Cook’s distance değeri
de çok küçük olduğunda küçük bir fark bile önem taşıyablir.

``` r
olcut1Index<-which(dist>olcut1)
olcut2Index<-which(dist>olcut2)
```

``` r
length(olcut1Index)
```

    ## [1] 1370

``` r
length(olcut2Index)
```

    ## [1] 759

olcut1’e göre 1358, olcut2’ye göre de 763 tane aykırı değerin var olduğu
tespit edilmiştir. Bu noktada aralarından bir tanesi seçilerek model
oluşturmak gerekiyor. olcut1’i seçelim. Görsel olarak Cook’s distance’ı
inceleyelim. (Ölçüt başına düşen aykırı değer değişiyor fakat nereye
seed koymam gerektiğini bulamadım.)

![](Çoklu-Doğrusal-Regresyon-ile-Tarımsal-Verim-Analizi_files/figure-gfm/cooks-1.png)<!-- -->

Cook’s distance çizgisinin üzerinde kalan aykırı değerleri trainset’ten
çıkaralım.

``` r
trainsetrem<-trainset[-olcut1Index,]
```

``` r
nrow(trainset)
```

    ## [1] 16000

``` r
nrow(trainsetrem)
```

    ## [1] 14630

## Model Karşılaştırması
<sup>[İçeriğe dön.](#İçerik)</sup>

Aykırı değerlerden arınmış değerlerle yeni bir model oluşturalım.

``` r
model2<-lm(Yield_kg_per_hectare ~ Soil_Quality+Rainfall_mm+Fertilizer_Amount_kg_per_hectare+
             Sunny_Days, data=trainsetrem)
```

model1 ve model2’yi karşılaştıralım.

``` r
summary(model1)
```

    ## 
    ## Call:
    ## lm(formula = Yield_kg_per_hectare ~ Soil_Quality + Rainfall_mm + 
    ##     Fertilizer_Amount_kg_per_hectare + Sunny_Days, data = trainset)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -637.07 -125.05   22.81  129.79  669.54 
    ## 
    ## Coefficients:
    ##                                   Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                      523.19497   18.37042   28.48   <2e-16 ***
    ## Soil_Quality                       1.61231    0.10063   16.02   <2e-16 ***
    ## Rainfall_mm                       -0.49375    0.01457  -33.89   <2e-16 ***
    ## Fertilizer_Amount_kg_per_hectare   0.78981    0.02027   38.97   <2e-16 ***
    ## Sunny_Days                         1.78963    0.14631   12.23   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 184.5 on 15995 degrees of freedom
    ## Multiple R-squared:   0.16,  Adjusted R-squared:  0.1598 
    ## F-statistic: 761.5 on 4 and 15995 DF,  p-value: < 2.2e-16

``` r
summary(model2)
```

    ## 
    ## Call:
    ## lm(formula = Yield_kg_per_hectare ~ Soil_Quality + Rainfall_mm + 
    ##     Fertilizer_Amount_kg_per_hectare + Sunny_Days, data = trainsetrem)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -492.27 -105.44   17.81  115.37  465.56 
    ## 
    ## Coefficients:
    ##                                   Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                      519.76174   16.94957   30.66   <2e-16 ***
    ## Soil_Quality                       1.65260    0.09138   18.09   <2e-16 ***
    ## Rainfall_mm                       -0.49611    0.01352  -36.70   <2e-16 ***
    ## Fertilizer_Amount_kg_per_hectare   0.77949    0.01843   42.30   <2e-16 ***
    ## Sunny_Days                         1.93776    0.13585   14.26   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 158.2 on 14625 degrees of freedom
    ## Multiple R-squared:  0.1992, Adjusted R-squared:  0.1989 
    ## F-statistic: 909.3 on 4 and 14625 DF,  p-value: < 2.2e-16

model2’nin ***Residual standard error*** değeri model1’inkinden
küçüktür. model2’nin $R^2$ değeri model1’inkinden büyüktür.
model2, model1’den daha iyi görünmektedir.

``` r
AIC(model1,k=6)
```

    ## [1] 212407.2

``` r
AIC(model2,k=6)
```

    ## [1] 189714.9

model2’nin AIC değeri model1’inkinden küçüktür. model2, model1’den daha
iyidir.

``` r
BIC(model1)
```

    ## [1] 212429.2

``` r
BIC(model2)
```

    ## [1] 189736.4

model2’nin BIC değeri model1’inkinden küçüktür. model2, model1’den daha
iyidir.

model1’in $R^2$ değerinin daha iyi çıkmasının nedeni gözlem
sayısının fazlalığından kaynaklı olabilir. Gözlem sayısı arttıkça
$R^2$ değeri yükselme eğilimindedir.

## Test Seti üzerinden model değerlendirme
<sup>[İçeriğe dön.](#İçerik)</sup>

``` r
predictions1<-predict(model1, testset)
predictions2<-predict(model2, testset)
```

``` r
R2(predictions1, testset$Yield_kg_per_hectare)
```

    ## [1] 0.1887379

``` r
R2(predictions2, testset$Yield_kg_per_hectare)
```

    ## [1] 0.1886366

model2’nin $R^2$ değeri model1’inkinden küçüktür. model1,
model2’den çok az da olsa daha iyidir.

``` r
RMSE(predictions1,testset$Yield_kg_per_hectare)
```

    ## [1] 183.103

``` r
RMSE(predictions2,testset$Yield_kg_per_hectare)
```

    ## [1] 183.5424

model2’nin ***RMSE*** değeri model1’inkinden büyüktür. model1,
model2’den çok az da olsa daha iyidir.

``` r
MAE(predictions1,testset$Yield_kg_per_hectare)
```

    ## [1] 147.1683

``` r
MAE(predictions2,testset$Yield_kg_per_hectare)
```

    ## [1] 146.5574

model2’nin ***MAE*** değeri model1’inkinden küçüktür. model2, model1’den
çok az da olsa daha iyidir.

``` r
par(mfrow=c(1,2))
plot(model2)
```

![](Çoklu-Doğrusal-Regresyon-ile-Tarımsal-Verim-Analizi_files/figure-gfm/plot_model2-1.png)<!-- -->![](Çoklu-Doğrusal-Regresyon-ile-Tarımsal-Verim-Analizi_files/figure-gfm/plot_model2-2.png)<!-- -->

Grafiklere karşılaştırdığımızda da model1, model2’den daha iyi
görünmektedir.

Tahmin sonuçlarına, AIC ve BIC değerlerine bakıldığında model2 daha iyi
bir performans sergilese de test seti ile yapılan performans
kıyaslamasında model1 daha başarılı olmuştur. AIC ve BIC değerleri
arasında da çok büyük farklar yoktur fakat model1’in grafikleri çok daha
iyi durmaktadır. Bunun sebebi verinin çok düzenli olması olabilir diye
düşünüyorum. Aykırı değerleri atarak oluşturduğumuz modelin bir tık daha
kötü performans sergilemesini beklememiştim. Overfitting olma durumunu
düşünüyorum fakat performans farkı çok büyük değil ve model2 grafikleri
daha kötü. Bu noktada belki k_fold cross validation yapılıp sonuçlar
yeniden değerlendirilebilir.

## Çoklu Bağlantı Sorunu (Multicollinearity)
<sup>[İçeriğe dön.](#İçerik)</sup>

Bağımsız değişkenlerin birbirleriyle yüksek dereceli ilşkileri varsa
çoklu bağlantı sorunu(multicollinearity) ile karşılaşılabilir. İlk
aşamada uyguladığımız korelasyon testinden yüksek dereceli bir ilişki
olmadığı anlaşılıyordu fakat diğer yöntemi de deneyelim.

``` r
vif(model2)
```

    ##                     Soil_Quality                      Rainfall_mm 
    ##                         1.000337                         1.000186 
    ## Fertilizer_Amount_kg_per_hectare                       Sunny_Days 
    ##                         1.000397                         1.000262

VIF değerleri 10’un çok altında bir değere sahip. Çoklu bağlantı
sorunu(multicollinearity) yoktur.

## İlişkili Hatalar
<sup>[İçeriğe dön.](#İçerik)</sup>

Doğrusal regresyon modellerinde en küçük kareler yönteminin başarılı bir
biçimde uygulanıp istenilen sonuçları verebilmesi için bazı
varsayımların sağlanması gerekir. Bu varsayımlardan bir tanesi de hata
terimi değerleri arasında otokorelasyon bulunmamasıdır. Bu varsayımın
sağlanmaması durumunda en küçük kareler yöntemiyle bulunan tahminler
sapmalı, t ve F gibi testler olduğundan büyük çıkıp güvenilirliğini
yitirirler.

Eğer hatalar arasında ilişki yoksa hataların ε=0 doğrusu etrafında
rastgele dağılması gerekir.
![](Çoklu-Doğrusal-Regresyon-ile-Tarımsal-Verim-Analizi_files/figure-gfm/otocorr-1.png)<!-- -->

Bu grafikten otokorelasyon sorunu olmadığı açık bir şekilde
görülmektedir.

İstatistiksel olarak anlamlılığını inceleyebiliriz.

``` r
summary(lm(tail(residuals(model2),n-1) ~ head(residuals(model2),n-1) -1))
```

    ## 
    ## Call:
    ## lm(formula = tail(residuals(model2), n - 1) ~ head(residuals(model2), 
    ##     n - 1) - 1)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -493.39 -105.53   18.02  115.17  467.79 
    ## 
    ## Coefficients:
    ##                                Estimate Std. Error t value Pr(>|t|)
    ## head(residuals(model2), n - 1) 0.012635   0.008267   1.528    0.126
    ## 
    ## Residual standard error: 158.1 on 14628 degrees of freedom
    ## Multiple R-squared:  0.0001597,  Adjusted R-squared:  9.133e-05 
    ## F-statistic: 2.336 on 1 and 14628 DF,  p-value: 0.1264

Beklentimiz bu modelin anlamlı olmamasıdır. Model p-value değeri
0.05’ten oldukça büyüktür. *Model anlamlı değildir.*

Durbin-Watson test istatistiğini kullanalım.

``` r
dwtest(Yield_kg_per_hectare ~ Soil_Quality+Rainfall_mm+Fertilizer_Amount_kg_per_hectare+
         Sunny_Days, data=trainsetrem)
```

    ## 
    ##  Durbin-Watson test
    ## 
    ## data:  Yield_kg_per_hectare ~ Soil_Quality + Rainfall_mm + Fertilizer_Amount_kg_per_hectare +     Sunny_Days
    ## DW = 1.9745, p-value = 0.06146
    ## alternative hypothesis: true autocorrelation is greater than 0

Hipotez, ‘ $H_0$:Hatalar arasında korelasyon yoktur.’ şeklinde
kurulur. *p-value* değerlendirildiğinde $H_0$ hipotezi rededdilemez.
Yani hatalar arasında korelasyon olmadığı görülür. Hesaplanan d, 0 ile 4
arasında değer almaktadır. d değeri 2’ye yakın olduğundan otokorelasyon
olmadığına işaret eder.

Breusch-Godfrey Test’ini kullanalım.

``` r
lmtest::bgtest(model2, order = 3)
```

    ## 
    ##  Breusch-Godfrey test for serial correlation of order up to 3
    ## 
    ## data:  model2
    ## LM test = 3.4951, df = 3, p-value = 0.3214

Hipotez, ‘ $H_0$:Hatalar arasında korelasyon yoktur.’ şeklinde
kurulur. *p-value* değerlendirildiğinde $H_0$ hipotezi rededdilemez.
Yani hatalar arasında korelasyon olmadığı görülür.
