from flask import Flask, render_template,request, jsonify,json
import bs4 as bs
import urllib.request
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from gensim.models import Word2Vec

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/sample",methods=["GET","POST"])
def sample():
    name = request.form["search"]
    nltk.download('stopwords')
    scrapped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Artificial_intelligence')
    article = scrapped_data .read()

    parsed_article = bs.BeautifulSoup(article,'lxml')

    paragraphs = parsed_article.find_all('p')

    article_text = ""

    for p in paragraphs:
        article_text += p.text
    processed_article = article_text.lower()
    processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )
    processed_article = re.sub(r'\s+', ' ', processed_article)
    all_sentences = nltk.sent_tokenize(processed_article)

    all_words = [nltk.word_tokenize(sent) for sent in all_sentences]
    for i in range(len(all_words)):
        all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]
    word2vec = Word2Vec(all_words, min_count=2)
    vocabulary = word2vec.wv.vocab
    
    v1 = word2vec.wv[name]

    sim_words = word2vec.wv.most_similar(name)
    # data = {'v1':v1}
    # data = data.jsonify(data)
    # sp = sim_words.split()
    
    return render_template("display.html",sim_words=dict(sim_words))

if __name__ == "__main__":
    app.run('0.0.0.0', 8085)