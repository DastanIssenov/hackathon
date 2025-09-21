# Auto-converted from Jupyter Notebook to Streamlit
# Tip: run with `streamlit run app.py`
import streamlit as st

st.set_page_config(page_title="Converted Notebook", layout="wide")

# --- Notebook compatibility helpers ---
# Redirect common notebook display calls to Streamlit
def display(*args, **kwargs):
    for a in args:
        st.write(a)

# Matplotlib show() -> Streamlit pyplot
try:
    import matplotlib.pyplot as plt
    def _st_show(*args, **kwargs):
        st.pyplot(plt.gcf())
    plt.show = _st_show
except Exception as _e:
    pass

# Optional: pandas dataframe pretty-printing goes to Streamlit
try:
    import pandas as pd  # noqa: F401
    import numpy as np    # noqa: F401
except Exception:
    pass

st.markdown("### 📓 Notebook: *new_note.ipynb* (converted)", help="This page was generated automatically from your .ipynb file.")


# --- [Code cell 1] ---
with st.expander('Show / hide code', expanded=False):
    st.code('''import os
import pickle
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


load_dotenv()

GOOGLE_API_KEY = "AIzaSyA61tZZ4jWe35CYdkty-SESm_HMOqvJ1BY"
OPEN_AI_API_KEY = "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"

INDEX_PATH_TEMP = "index_temp.faiss"
METADATA_PATH_TEMP = "metadata_temp.pkl"

INDEX_PATH_BAN = "index_ban.faiss"
METADATA_PATH_BAN = "metadata_ban.pkl"

INDEX_PATH_CLAS = "index_clas.faiss"
METADATA_PATH_CLAS = "metadata_clas.pkl"

INDEX_PATH_LANG = "index_lang.faiss"
METADATA_PATH_LANG = "metadata_lang.pkl"


model_name = "intfloat/multilingual-e5-large"
model = SentenceTransformer(model_name)

index_clas = faiss.read_index(INDEX_PATH_CLAS)

with open(METADATA_PATH_CLAS, "rb") as f:
    metadata_clas = pickle.load(f)
    

index_temp = faiss.read_index(INDEX_PATH_TEMP)

with open(METADATA_PATH_TEMP, "rb") as f:
    metadata_temp = pickle.load(f)


index_lang = faiss.read_index(INDEX_PATH_LANG)

with open(METADATA_PATH_LANG, "rb") as f:
    metadata_lang = pickle.load(f)


index_ban = faiss.read_index(INDEX_PATH_BAN)

with open(METADATA_PATH_BAN, "rb") as f:
    metadata_ban = pickle.load(f)


def get_query_embedding(query):
    return model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

def search_document(query, index, metadata, k = 1):
    query_vector = get_query_embedding(query)
    distances, indices = index.search(query_vector, k)

    results = []
    for i, idx in enumerate(indices[0]):
        if distances[0][i] >= 0.3:
            results.append(metadata[idx])
    return results



import googleapiclient.discovery
import pandas as pd
from urllib.parse import urlparse, parse_qs

def get_video_id(url: str) -> str:
    \"\"\"
    Extract the video ID from a YouTube URL.
    \"\"\"
    parsed_url = urlparse(url)

    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query)["v"][0]
    

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]
    
    raise ValueError("Invalid YouTube URL")

path = input("Enter path to instagram post or youtube video: ")

''', language='python')
# Executing cell code:
import os
import pickle
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


load_dotenv()

GOOGLE_API_KEY = "AIzaSyA61tZZ4jWe35CYdkty-SESm_HMOqvJ1BY"
OPEN_AI_API_KEY = "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"


INDEX_PATH_TEMP = "index_temp.faiss"
METADATA_PATH_TEMP = "metadata_temp.pkl"

INDEX_PATH_BAN = "index_ban.faiss"
METADATA_PATH_BAN = "metadata_ban.pkl"

INDEX_PATH_CLAS = "index_clas.faiss"
METADATA_PATH_CLAS = "metadata_clas.pkl"

INDEX_PATH_LANG = "index_lang.faiss"
METADATA_PATH_LANG = "metadata_lang.pkl"

# INDEX_PATH_TEMP = "/Users/dastanissenov/Desktop/datathon/classify/index_temp.faiss"
# METADATA_PATH_TEMP = "/Users/dastanissenov/Desktop/datathon/classify/metadata_temp.pkl"

# INDEX_PATH_BAN = "/Users/dastanissenov/Desktop/datathon/classify/index_ban.faiss"
# METADATA_PATH_BAN = "/Users/dastanissenov/Desktop/datathon/classify/metadata_ban.pkl"

# INDEX_PATH_CLAS = "/Users/dastanissenov/Desktop/datathon/classify/index_clas.faiss"
# METADATA_PATH_CLAS = "/Users/dastanissenov/Desktop/datathon/classify/metadata_clas.pkl"

# INDEX_PATH_LANG = "/Users/dastanissenov/Desktop/datathon/classify/index_lang.faiss"
# METADATA_PATH_LANG = "/Users/dastanissenov/Desktop/datathon/classify/metadata_lang.pkl"


model_name = "intfloat/multilingual-e5-large"
model = SentenceTransformer(model_name)

index_clas = faiss.read_index(INDEX_PATH_CLAS)

with open(METADATA_PATH_CLAS, "rb") as f:
    metadata_clas = pickle.load(f)
    

index_temp = faiss.read_index(INDEX_PATH_TEMP)

with open(METADATA_PATH_TEMP, "rb") as f:
    metadata_temp = pickle.load(f)


index_lang = faiss.read_index(INDEX_PATH_LANG)

with open(METADATA_PATH_LANG, "rb") as f:
    metadata_lang = pickle.load(f)


index_ban = faiss.read_index(INDEX_PATH_BAN)

with open(METADATA_PATH_BAN, "rb") as f:
    metadata_ban = pickle.load(f)


def get_query_embedding(query):
    return model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

def search_document(query, index, metadata, k = 1):
    query_vector = get_query_embedding(query)
    distances, indices = index.search(query_vector, k)

    results = []
    for i, idx in enumerate(indices[0]):
        if distances[0][i] >= 0.3:
            results.append(metadata[idx])
    return results



import googleapiclient.discovery
import pandas as pd
from urllib.parse import urlparse, parse_qs

def get_video_id(url: str) -> str:
    """
    Extract the video ID from a YouTube URL.
    """
    parsed_url = urlparse(url)

    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query)["v"][0]
    

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]
    
    raise ValueError("Invalid YouTube URL")

path = input("Enter path to instagram post or youtube video: ")



# --- [Code cell 2] ---
with st.expander('Show / hide code', expanded=False):
    st.code('''
try:
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyA61tZZ4jWe35CYdkty-SESm_HMOqvJ1BY"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY
    )

    request = youtube.commentThreads().list(
        part="snippet",
        videoId= get_video_id(path),
        maxResults=200
    )
    response = request.execute()

    comments = []

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append([
            comment['authorDisplayName'],
            comment['publishedAt'],
            comment['textDisplay']
        ])

    df = pd.DataFrame(comments, columns=['author', 'posted_at', 'text'])

except:
    from selenium import webdriver
    from bs4 import BeautifulSoup
    import time


    df = pd.DataFrame(columns = ['author', 'updated_at', 'text'])
    driver = webdriver.Chrome()

    driver.get(path)

    time.sleep(5)
    html = driver.page_source

    soup = BeautifulSoup(html, "html.parser")
    divs = soup.find_all("div", class_ = "html-div xdj266r x14z9mp xat24cr x1lziwak xyri2b x1c1uobl x9f619 xjbqb8w x78zum5 x15mokao x1ga7v0g x16uus16 xbiv7yw xsag5q8 xz9dl7a x1uhb9sk x1plvlek xryxfnj x1c4vz4f x2lah0s x1q0g3np xqjyukv x1qjc9v5 x1oa3qoh x1nhvcw1")
    print(divs)

    comments = []

    for div in divs[1:]:
        spans = div.find_all("span")

        comments.append([
            spans[1].get_text(strip=True),
            spans[2].get_text(strip=True),
            spans[3].get_text(strip=True)
        ])
    df = pd.DataFrame(comments, columns=['author', 'posted_at', 'text'])
    driver.quit()
''', language='python')
# Executing cell code:

try:
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyA61tZZ4jWe35CYdkty-SESm_HMOqvJ1BY"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY
    )

    request = youtube.commentThreads().list(
        part="snippet",
        videoId= get_video_id(path),
        maxResults=200
    )
    response = request.execute()

    comments = []

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append([
            comment['authorDisplayName'],
            comment['publishedAt'],
            comment['textDisplay']
        ])

    df = pd.DataFrame(comments, columns=['author', 'posted_at', 'text'])

except:
    from selenium import webdriver
    from bs4 import BeautifulSoup
    import time


    df = pd.DataFrame(columns = ['author', 'updated_at', 'text'])
    driver = webdriver.Chrome()

    driver.get(path)

    time.sleep(5)
    html = driver.page_source

    soup = BeautifulSoup(html, "html.parser")
    divs = soup.find_all("div", class_ = "html-div xdj266r x14z9mp xat24cr x1lziwak xyri2b x1c1uobl x9f619 xjbqb8w x78zum5 x15mokao x1ga7v0g x16uus16 xbiv7yw xsag5q8 xz9dl7a x1uhb9sk x1plvlek xryxfnj x1c4vz4f x2lah0s x1q0g3np xqjyukv x1qjc9v5 x1oa3qoh x1nhvcw1")
    print(divs)

    comments = []

    for div in divs[1:]:
        spans = div.find_all("span")

        comments.append([
            spans[1].get_text(strip=True),
            spans[2].get_text(strip=True),
            spans[3].get_text(strip=True)
        ])
    df = pd.DataFrame(comments, columns=['author', 'posted_at', 'text'])
    driver.quit()


# --- [Code cell 3] ---
with st.expander('Show / hide code', expanded=False):
    st.code('''
# df

INDEX_PATH_Q = "/Users/dastanissenov/Desktop/datathon/embedded/index.faiss"
METADATA_PATH_Q = "/Users/dastanissenov/Desktop/datathon/embedded/metadata.pkl"
INDEX_PATH_A = "/Users/dastanissenov/Desktop/datathon/embedded/index_a.faiss"
METADATA_PATH_A = "/Users/dastanissenov/Desktop/datathon/embedded/metadata_a.pkl"


index_q = faiss.read_index(INDEX_PATH_Q)

with open(METADATA_PATH_A, "rb") as f:
    metadata_a = pickle.load(f)

import requests

url = "https://api.openai.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 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"
}


lang = []
temp = []
to_ban = []
clas = []
answers = []
for i in df.iterrows():
    
    data = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": "определи язык комментария: русский или казахский. Ответ должен состоять просто из одного слова"},
            {"role": "user", "content": f"Комментарий: {i[1]['text']}"}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    otvet = response.json()
    lang.append(search_document(otvet["choices"][0]['message']['content'], index_lang, metadata_lang)[0])

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Классифицируй комментарий как одну из этих категорий: вопрос, отзыв, жалоба, благодарность или неизвестный. Ответ должен быть одим из этих слов на русском. Если есть знак вопроса то это скорее всего вопрос"},
            {"role": "user", "content": f"Комментарий: {i[1]['text']}"}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    otvet = response.json()
    clas.append(search_document(otvet["choices"][0]['message']['content'], index_clas, metadata_clas)[0])

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "определи тон комментария: позитивный, негативный, нейтральный. Ответ должен быть одим из этих слов на русском"},
            {"role": "user", "content": f"Комментарий: {i[1]['text']}"}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    otvet = response.json()
    temp.append(search_document(otvet["choices"][0]['message']['content'], index_temp, metadata_temp)[0])

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "относиться ли комментарий к одной из этих групп: спам, оскорбления, нецензурная лексика. Если да то ответ должен быть одим из этих слов на русском если нет то ответ нет"},
            {"role": "user", "content": f"Комментарий: {i[1]['text']}"}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    otvet = response.json()
    to_ban.append(search_document(otvet["choices"][0]['message']['content'], index_ban, metadata_ban)[0])

    if clas[-1] == "вопрос":
        comment = i[1]['text']  
        chunks = search_document(comment, index_q, metadata_a, k = 3)
        data = {
            "model": "gpt-4.1-mini",
            "messages": [
                {"role": "system", "content": "Определи вопрос и ответь на него основываясь на этих данных: " + " ".join(chunks) + ". Ответь на этом языке " + lang[-1]},
                {"role": "user", "content": f"Вопрос: {comment}"}
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        otvet = response.json()
        answers.append(otvet["choices"][0]['message']['content'])
    else:
        answers.append("")


df['lang'] = lang
df['clas'] = clas
df['temp'] = temp
df['to_ban'] = to_ban
df['answer'] = answers
df.to_csv("comments_classified.csv", index = False)''', language='python')
# Executing cell code:

# df

INDEX_PATH_Q = "/Users/dastanissenov/Desktop/datathon/embedded/index.faiss"
METADATA_PATH_Q = "/Users/dastanissenov/Desktop/datathon/embedded/metadata.pkl"
INDEX_PATH_A = "/Users/dastanissenov/Desktop/datathon/embedded/index_a.faiss"
METADATA_PATH_A = "/Users/dastanissenov/Desktop/datathon/embedded/metadata_a.pkl"


index_q = faiss.read_index(INDEX_PATH_Q)

with open(METADATA_PATH_A, "rb") as f:
    metadata_a = pickle.load(f)

import requests

url = "https://api.openai.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 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"
}


lang = []
temp = []
to_ban = []
clas = []
answers = []
for i in df.iterrows():
    
    data = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": "определи язык комментария: русский или казахский. Ответ должен состоять просто из одного слова"},
            {"role": "user", "content": f"Комментарий: {i[1]['text']}"}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    otvet = response.json()
    lang.append(search_document(otvet["choices"][0]['message']['content'], index_lang, metadata_lang)[0])

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Классифицируй комментарий как одну из этих категорий: вопрос, отзыв, жалоба, благодарность или неизвестный. Ответ должен быть одим из этих слов на русском. Если есть знак вопроса то это скорее всего вопрос"},
            {"role": "user", "content": f"Комментарий: {i[1]['text']}"}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    otvet = response.json()
    clas.append(search_document(otvet["choices"][0]['message']['content'], index_clas, metadata_clas)[0])

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "определи тон комментария: позитивный, негативный, нейтральный. Ответ должен быть одим из этих слов на русском"},
            {"role": "user", "content": f"Комментарий: {i[1]['text']}"}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    otvet = response.json()
    temp.append(search_document(otvet["choices"][0]['message']['content'], index_temp, metadata_temp)[0])

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "относиться ли комментарий к одной из этих групп: спам, оскорбления, нецензурная лексика. Если да то ответ должен быть одим из этих слов на русском если нет то ответ нет"},
            {"role": "user", "content": f"Комментарий: {i[1]['text']}"}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    otvet = response.json()
    to_ban.append(search_document(otvet["choices"][0]['message']['content'], index_ban, metadata_ban)[0])

    if clas[-1] == "вопрос":
        comment = i[1]['text']  
        chunks = search_document(comment, index_q, metadata_a, k = 3)
        data = {
            "model": "gpt-4.1-mini",
            "messages": [
                {"role": "system", "content": "Определи вопрос и ответь на него основываясь на этих данных: " + " ".join(chunks) + ". Ответь на этом языке " + lang[-1]},
                {"role": "user", "content": f"Вопрос: {comment}"}
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        otvet = response.json()
        answers.append(otvet["choices"][0]['message']['content'])
    else:
        answers.append("")


df['lang'] = lang
df['clas'] = clas
df['temp'] = temp
df['to_ban'] = to_ban
df['answer'] = answers
df.to_csv("comments_classified.csv", index = False)
