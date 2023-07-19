
#reviewerID Kullanıcı ID’si
#asin Ürün ID’si
#reviewerName Kullanıcı Adı
#helpful Faydalı değerlendirme derecesi
#reviewText Değerlendirme
#overall Ürün rating’i
#summary Değerlendirme özeti
#unixReviewTime Değerlendirme zamanı
#reviewTime Değerlendirme zamanı Raw
#day_diff Değerlendirmeden itibaren geçen gün sayısı
#helpful_yes Değerlendirmenin faydalı bulunma sayısı
#total_vote Değerlendirmeye verilen oy sayısı
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 10)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


df = pd.read_csv("C:/Users/aligi/OneDrive/Masaüstü/rating product case study/amazon_review.csv")

df.head()

#Ürünün ortalama puanını hesaplayınız.
#Tarihe göre ağırlıklı puan ortalamasını hesaplayınız.
#Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız
df["overall"].mean()

df.describe().T
df["day_diff"].quantile(0.50)
df.loc[df["day_diff"] <=50, 'overall'].mean()
df.loc[(df["day_diff"] > 50) &(df["day_diff"] <=150),"overall"].mean()

df.loc[df["day_diff"] <= df["day_diff"].quantile(0.25), "overall"].mean()
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.25)) & (df["day_diff"] <= df["day_diff"].quantile(0.50)), "overall"].mean()
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.50)) & (df["day_diff"] <= df["day_diff"].quantile(0.75)), "overall"].mean()
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.75)), "overall"].mean()

def time_based_weighted_average(dataframe, w1=40, w2=30, w3=20, w4=10):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.25), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.25)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.50)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.50)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w4 / 100

time_based_weighted_average(df)
time_based_weighted_average(df,28,26,24,22)
# helpful_no değişkenini üretiniz.
#score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz
#20 Yorumu belirleyiniz ve sonuçları Yorumlayınız.
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head(20)
df.sort_values("helpful_no", ascending=False).head(20)
df.sort_values("helpful_yes", ascending=False).head(20)
def score_up_down_diff(up, down):
    return up - down

score_up_down_diff(600,400)

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

score_average_rating(600,400)

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head(10)
df.tail(20)
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head()

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head()
df.sort_values("wilson_lower_bound", ascending=False).head(20)