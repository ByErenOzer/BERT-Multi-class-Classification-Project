import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Veri setini yükler, 'clean_text' ve 'sentiment' sütunlarındaki NaN değerlerini siler.
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    df.dropna(subset=['clean_text'], inplace=True)
    df.dropna(subset=['sentiment'], inplace=True)
    return df

 # Metinden emojileri ve URL'leri kaldırır.
def remove_emojis_and_urls(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002500-\U00002BEF"  
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", flags=re.UNICODE)
    url_pattern = re.compile(r'(https?://\S+)|(www\.\S+)|([a-zA-Z0-9./]+\.[a-zA-Z]{2,})')
    text = emoji_pattern.sub(r'', text)
    text = url_pattern.sub(r'', text)
    return text

 # Metinden @bahsedenleri ve #hashtagleri kaldırır.
def remove_mentions_and_hashtags(text):
    return re.sub(r'[@#]\w+', '', text)

# DataFrame içindeki metinleri, belirlenen kısaltma sözlüğüne göre genişletir.
def preprocess_dataframe(df):
    abbreviations_dict = {
        "n'olur": "ne olur",
        "bi'": "bir",
        "Pzt.": "Pazartesi",
        "pzt.": "Pazartesi",
        "bsk.": "başkan",
        "slm.": "selam",
        "gbi": "gibi",
        "vs.": "vesaire",
        "sn.": "sayın",  # ya da "sayın", bağlama göre değişebilir
        "dk.": "dakika",
        "çk.": "çok",
        "grş.": "görüş",
        "tlfn": "telefon",
        "msg": "mesaj",
        "hst.": "hasta",
        "bşk.": "başkan",
        "fkr.": "fikir",
        "örn.": "örnek",
        "mrb.": "merhaba",
        "tl.": "Türk Lirası",
        "dğl.": "değil",
        "dr.": "doktor",
        "prof.": "profesör",
        "mh.": "mahalle",
        "sk.": "sokak",
        "blv.": "bulvar",
        "cd.": "cadde",
        "apt.": "apartman",
        "dpt.": "departman",
        "ünv.": "üniversite",
        "gn.": "genel",
        "fak.": "fakülte",
        "böl.": "bölüm",
        "no.": "numara",
        "abv ": "allah belanı versin ",
        "ibb.": "istanbul belediye başkanı"
    }
    df['clean_text'] = df['clean_text'].apply(lambda x: expand_abbreviations(x, abbreviations_dict))
    return df

# Metindeki kısaltmaları tam halleriyle değiştirir.
def expand_abbreviations(text, abbreviations_dict):
    for abbrev, full_form in abbreviations_dict.items():
        text = text.replace(abbrev, full_form)
    return text

# tüm sayıları df den kaldırır.
def remove_all_numbers(text):
    # Metinden tüm sayıları kaldırır.
    return re.sub(r'\d+', '', text)

# Metni küçük harfe çevirir, emojileri, URL'leri, bahsedilenleri, hashtagleri ve noktalama işaretlerini kaldırır, fazla boşlukları siler.
def apply_text_normalizations(df, text_col='clean_text'):
    df['clean_text'] = df[text_col].apply(lambda x: x.lower())
    df['clean_text'] = df['clean_text'].apply(remove_emojis_and_urls)
    df['clean_text'] = df['clean_text'].apply(remove_mentions_and_hashtags)
    df['clean_text'] = df['clean_text'].apply(remove_all_numbers)  # Uygulanan her bir satır için bu fonksiyonu çağır
    df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))  # Noktalama işaretlerini kaldır
    df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())  # Fazla boşlukları temizle
    return df


if __name__ == "__main__":
    file_path = 'C:\\Users\\Eren\\PycharmProjects\\pythonProject15\\df8000veri_son.csv'
    df = load_data(file_path)
    df = apply_text_normalizations(df)
    df = preprocess_dataframe(df)
    print("Preprocessing completed.")
    df.to_csv('C:\\Users\\Eren\\PycharmProjects\\pythonProject15\\cleaned1_data.csv', index=False)