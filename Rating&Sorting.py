###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı


###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/amazon_review.csv")
df.head()
df.shape

# Ürünün ortalama puanı
df["overall"].mean()

df["reviewTime"].max()

###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
###################################################

df["reviewTime"] = pd.to_datetime(df["reviewTime"])
df['reviewTime'].dtypes  # Type bilgisini aldık

current_date = pd.to_datetime('2014-12-10 0:0:0')

df["days"] = (current_date - df["reviewTime"]).dt.days

df.loc[df["days"] <= 30, "overall"].mean()  # Son 30 gün

df.loc[(df["days"] > 30) & (df["days"] <= 90), "overall"].mean()  # 30. ve 90.günlerin arasındakiler

df.loc[(df["days"] > 90) & (df["days"] <= 180), "overall"].mean()  # 90. ve 180. günleri arasındakiler

df.loc[(df["days"] > 180), "overall"].mean()  # 180 günden sorankiler

df.loc[df["days"] <= 30, "overall"].mean() * 28 / 100 + \
df.loc[(df["days"] > 30) & (df["days"] <= 90), "overall"].mean() * 26 / 100 + \
df.loc[(df["days"] > 90) & (df["days"] <= 180), "overall"].mean() * 24 / 100 + \
df.loc[(df["days"] > 180), "overall"].mean() * 22 / 100


def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= 30, "overall"].mean() * w1 / 100 + \
        dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "overall"].mean() * w2 / 100 + \
        dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "overall"].mean() * w3 / 100 + \
        dataframe.loc[(dataframe["days"] > 180), "overall"].mean() * w4 / 100


time_based_weighted_average(df)

###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################


###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]


###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################

# score_pos_neg_diff
def score_up_down_diff(helpful_yes, helpful_no):
    return helpful_yes - helpful_no


# score_average_rating
def score_average_rating(helpful_yes, helpful_no):
    if helpful_yes + helpful_no == 0:
        return 0
    return helpful_yes / (helpful_yes + helpful_no)


# wilson_lower_bound
def wilson_lower_bound(helpful_yes, helpful_no, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = helpful_yes + helpful_no
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * helpful_yes / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


# df.loc[], belirli satırları veya sütunları seçmek için kullanılan bir indeksleme yöntemidir.
# Öte yandan, apply fonksiyonu ve lambda row ifadesi,
# tüm DataFrame'deki her bir satır için belirli bir işlevi çağırmak ve işlem yapmak için kullanılır.


# axis=0: İşlev sütunlar boyunca uygulanır. Yani, her bir sütun üzerinde işlem yapılır.
# axis=1: İşlev satırlar boyunca uygulanır. Yani, her bir satır üzerinde işlem yapılır.

# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda row: score_up_down_diff(row["helpful_yes"], row["helpful_no"]), axis=1)

# score_average_rating
df["score_average_rating"] = df.apply(lambda row: score_average_rating(row["helpful_yes"], row["helpful_no"]), axis=1)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda row: wilson_lower_bound(row["helpful_yes"], row["helpful_no"]), axis=1)


# Yukarıdaki diğer fonksiyonlar bir skor hesaplar; ancak bu skor, yorumların ne kadar güvenilir olduğu hakkında bilgi vermez.
# Yorum sayısı arttıkça veya faydalı/faydasız oyların oranı değiştikçe, bu skorun güvenilirliği değişebilir. Wilson Lower Bound,
# örneklem büyüklüğü ve başarı oranı gibi faktörleri dikkate alarak daha güvenilir bir skor sağlar.
# Örneğin, bir yoruma 200 faydalı ve 100 faydasız oy verilmişse, bu durumda Wilson Lower Bound kullanılarak bir güven aralığı hesaplanabilir.
# Bu güven aralığı, yorumun gerçekten faydalı olup olmadığını belirlemekte daha güvenilir bir tahmin sağlar.

##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)
