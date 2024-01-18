from flask import Flask, render_template, request, jsonify
from flask_session import Session
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import seaborn as sns
import re,nltk,json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from keras.models import load_model
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import string
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from places import scrape_clinic_and_rating,scrape_clinic_and_rating1
from scrape import scrape_and_run1,scrape_and_run2,scrape_and_run3,scrape_and_run4
import cv2
from flask import Flask, render_template, Response
import winsound
import speech_recognition as sr




file_name = "finalized_model.sav"
loaded_model = joblib.load(file_name)
loaded_model1 = pickle.load(open('BestModel.sav', 'rb'))
model = joblib.load('decision_tree_model.pkl')



app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET KEY"] = "tisha123"
Session(app)

df = pd.read_excel('final.xlsx')
#df = df[:3000]
df = df.dropna()
df = df.sample(frac = 1)


#lm  = WordNetLemmatizer()
nltk.download('wordnet')

def text_transformation(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}


def text_cleaner(text):
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    newString = re.sub('[m]{2,}', 'mm', newString)
    return newString

df['cleaned'] = df["text"].apply(text_cleaner)

# Feature Extraction
X = df.cleaned
y = df.label
vect = CountVectorizer(max_features = 20000 , lowercase=False , ngram_range=(1,2))
X_cv =vect.fit_transform(X).toarray()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

vc = cv2.VideoCapture(0)
count = 0
fontScale = 2
color = (0, 0, 255)
thickness = 10
gcount = 0


def generate_frames():
    global count, gcount
    while True:
        ret, frame = vc.read()
        if not ret:
            break

        img = frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if not len(faces):
            count += 1
            gcount = 0

        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) == 2:
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                count = 0
                gcount += 1

        cv2.putText(img, str(count), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img, str(gcount), (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)

        if count > 50 and count < 100:
            cv2.putText(img, "Please look at the screen", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        cv2.LINE_AA)
            winsound.PlaySound('beep.wav', winsound.SND_FILENAME)
        elif count > 100 and count>120:
            cv2.putText(img, "Uh,oh!! you might be suffering from adhd", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1, cv2.LINE_AA)
        if gcount > 10 and  gcount < 150:
            cv2.putText(img, "Yayy !!! keep going ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)
        if gcount > 150:
            cv2.putText(img, "Yayy !!! you are not suffering from adhd ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/',methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/mental',methods=['GET', 'POST'])
def mental():
    return render_template('mental.html')

@app.route('/calculate4', methods=['GET', 'POST'])
def calculate4():
    data = {
    'age': int(request.form['age']),
    'feeling.nervous': int(request.form['q1']),
    'panic': int(request.form['q2']),
    'breathing.rapidly': int(request.form['q3']),
    'sweating': int(request.form['q4']),
    'trouble.in.concentration': int(request.form['q5']),
    'trouble.sleeping': int(request.form['q6']),
    'trouble.with.work': int(request.form['q7']),
    'hopelessness': int(request.form['q8']),
    'anger': int(request.form['q9']),
    'over.react': int(request.form['q10']),
    'change.in.eating': int(request.form['q11']),
    'suicidal.thought': int(request.form['q12']),
    'feeling.tired': int(request.form['q13']),
    'close.friend': int(request.form['q14']),
    'social.media.addiction': int(request.form['q15']),
    'weight.gain': int(request.form['q16']),
    'introvert': int(request.form['q17']),
    'popping.up.stressful.memory': int(request.form['q18']),
    'nightmares': int(request.form['q19']),
    'avoids.people.or.activities': int(request.form['q20']),
    'feeling.negative': int(request.form['q21']),
    'trouble.concentrating': int(request.form['q22']),
    'blamming.yourself': int(request.form['q23']),
    'hallucinations': int(request.form['q24']),
    'repetitive.behaviour': int(request.form['q25']),
    'seasonally': int(request.form['q26']),
    'increased.energy': int(request.form['q27'])
}
    print(data)
    new_data = pd.DataFrame([data])
    predictions = model.predict(new_data)
    disorder_names = [
            'ADHD',
            'ASD',
            'Loneliness',
            'MDD',
            'OCD',
            'PDD',
            'PTSD',
            'anxiety',
            'bipolar',
            'eating disorder',
            'psychotic depression',
            'sleeping disorder'
        ]
    predicted_disorder = [disorder_names[i] for i, prediction in enumerate(predictions[0]) if prediction == 1]
    predicted_disorder_string = ', '.join(predicted_disorder)
    print("Predicted Disorders:", predicted_disorder_string)
    return render_template('mental.html',predicted_disorder_string=predicted_disorder_string, flag=True)

@app.route('/bipolar',methods=['GET', 'POST'])
def bipolar():
    return render_template('bipolar.html')

@app.route('/adhd',methods=['GET', 'POST'])
def adhd():
    return render_template('adhd.html')

@app.route('/depres',methods=['GET', 'POST'])
def depres():
    return render_template('depres.html')

@app.route('/ptsd',methods=['GET', 'POST'])
def ptsd():
    return render_template('ptsd.html')

@app.route('/contact',methods=['GET', 'POST'])
def contact():
    return render_template('contact.html')

@app.route('/blog',methods=['GET', 'POST'])
def blog():
    return render_template('blog.html')

@app.route('/depresTest',methods=['GET', 'POST'])
def depresTest():
    return render_template('depresTest.html')

@app.route('/calculate', methods=['GET', 'POST'])
def calculate():
    q1 = int(request.form['q1'])
    q2 = int(request.form['q2'])
    q3 = int(request.form['q3'])
    q4 = int(request.form['q4'])
    q5 = int(request.form['q5'])
    q6 = int(request.form['q6'])
    q7 = int(request.form['q7'])
    q8 = int(request.form['q8'])
    q9 = int(request.form['q9'])
    q10 = int(request.form['q10'])
    print(q9)
    print(q10)
    total_score = q1+q2+q3+q4+q5+q6+q7+q8+q9+q10  
    return render_template('depresTest.html', score=total_score)

@app.route('/depresAdvTest',methods=['GET', 'POST'])
def depresAdvTest():
    return render_template('depresAdvTest.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    flag=True
    score=0
    score1=0
    if 'audioFile' in request.files:
        audio_file = request.files['audioFile']
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as audio_file:
            recognizer.adjust_for_ambient_noise(audio_file)
            audio_data = recognizer.record(audio_file)
            q2 = recognizer.recognize_google(audio_data)
    else:
        q2 = request.form['q2']
    q1 = request.form['q1']
    q3 = request.form['q3']
    clean_q1 = text_cleaner(q1)
    clean_q2 = text_cleaner(q2)
    clean_q3 = text_cleaner(q3)
    print(clean_q1)
    print(clean_q2)
    print(clean_q3)
    result1 = loaded_model.predict([clean_q1])
    single_prediction1 = loaded_model1.predict(vect.transform([clean_q1]).toarray())[0]
    result2 = loaded_model.predict([clean_q2])
    single_prediction2 = loaded_model1.predict(vect.transform([clean_q2]).toarray())[0]
    result3 = loaded_model.predict([clean_q3])
    single_prediction3 = loaded_model1.predict(vect.transform([clean_q3]).toarray())[0]
    output = {"suicide":1.0, "non-suicide":0.0}
    print(single_prediction1)
    print(output[result1[0]])
    print(single_prediction2)
    print(output[result2[0]])
    print(single_prediction3)
    print(output[result3[0]])
    if(single_prediction1==1.0):
        score+=1
    if(single_prediction2==1.0):
        score+=1
    if(single_prediction3==1.0):
        score+=1
    if(result1=="suicide"):
        score1+=1
    if(result2=="suicide"):
        score1+=1
    if(result3=="suicide"):
        score1+=1
    print(score)
    print(score1)
    return render_template('depresAdvTest.html',score=score,score1=score1,flag=flag)


@app.route('/BipolarTest',methods=['GET', 'POST'])
def BipolarTest():
    return render_template('BipolarTest.html')

@app.route('/BipolarTest2',methods=['GET', 'POST'])
def BipolarTest2():
    return render_template('BipolarTest2.html')

@app.route('/BipolarTest3',methods=['GET', 'POST'])
def BipolarTest3():
    return render_template('BipolarTest3.html')

@app.route('/cal', methods=['GET', 'POST'])
def cal():
    q1 = int(request.form.get('q1'))
    q2 = int(request.form.get('q2'))
    q3 = int(request.form.get('q3'))
    q4 = int(request.form.get('q4'))
    q5 = int(request.form.get('q5'))
    q6 = int(request.form.get('q6'))
    q7 = int(request.form.get('q7'))
    q8 = int(request.form.get('q8'))
    q9 = int(request.form.get('q9'))
    q10 = int(request.form.get('q10'))
    q11 = int(request.form.get('q11'))
    print(q9)
    print(q10)
    total_score = q1+q2+q3+q4+q5+q6+q7+q8+q9+q10+q11  
    return render_template('BipolarTest2.html', score=total_score)

@app.route('/cal1', methods=['GET', 'POST'])
def cal1():
    q1 = int(request.form.get('q1'))
    q2 = int(request.form.get('q2'))
    q3 = int(request.form.get('q3'))
    q4 = int(request.form.get('q4'))
    q5 = int(request.form.get('q5'))
    q6 = int(request.form.get('q6'))
    q7 = int(request.form.get('q7'))
    q8 = int(request.form.get('q8'))
    q9 = int(request.form.get('q9'))
    q10 = int(request.form.get('q10'))
    q11 = int(request.form.get('q11'))
    q12 = int(request.form.get('q12'))
    q13 = int(request.form.get('q13'))
    q14 = int(request.form.get('q14'))
    q15 = int(request.form.get('q15'))
    q16 = int(request.form.get('q16'))
    print(q9)
    print(q10)
    total_score = q1+q2+q3+q4+q5+q6+q7+q8+q9+q10+q11+q12+q13+q14+q15+q16  
    return render_template('BipolarTest3.html', score=total_score)


@app.route('/calculate2', methods=['GET', 'POST'])
def calculate2():
    y_items = 0
    q1 = int(request.form['q1'])
    if q1 == 0:
        y_items += 1
    q2 = int(request.form['q2'])
    if q2 == 0:
        y_items += 1
    q3 = int(request.form['q3'])
    if q3 == 0:
        y_items += 1
    q4 = int(request.form['q4'])
    if q4 == 0:
        y_items += 1
    q5 = int(request.form['q5'])
    if q5 == 0:
        y_items += 1
    q6 = int(request.form['q6'])
    if q6 == 0:
        y_items += 1
    q7 = int(request.form['q7'])
    if q7 == 0:
        y_items += 1
    q8 = int(request.form['q8'])
    if q8 == 0:
        y_items += 1
    q9 = int(request.form['q9'])
    if q9 == 0:
        y_items += 1
    q10 = int(request.form['q10'])
    if q10 == 0:
        y_items += 1
    q11 = int(request.form['q11'])
    if q11 == 0:
        y_items += 1
    q12 = int(request.form['q12'])
    if q12 == 0:
        y_items += 1
    q13 = int(request.form['q13'])
    if q13 == 0:
        y_items += 1
    return render_template('BipolarTest.html', score=y_items)

@app.route('/PTSDTest',methods=['GET', 'POST'])
def PTSDTest():
    return render_template('PTSDTest.html')

@app.route('/calculate1', methods=['GET', 'POST'])
def calculate1():
    b_items = 0
    c_items = 0
    d_items = 0
    q1 = int(request.form['q1'])
    if q1 >= 3:
        b_items += 1
    q2 = int(request.form['q2'])
    if q2 >= 3:
        b_items += 1
    q3 = int(request.form['q3'])
    if q3 >= 3:
        b_items += 1
    q4 = int(request.form['q4'])
    if q4 >= 3:
        b_items += 1
    q5 = int(request.form['q5'])
    if q5 >= 3:
        b_items += 1
    q6 = int(request.form['q6'])
    if q6 >= 3:
        d_items += 1
    q7 = int(request.form['q7'])
    if q7 >= 3:
        c_items += 1
    q8 = int(request.form['q8'])
    if q8 >= 3:
        c_items += 1
    q9 = int(request.form['q9'])
    if q9 >= 3:
        c_items += 1
    q10 = int(request.form['q10'])
    if q10 >= 3:
        d_items += 1
    q11 = int(request.form['q11'])
    if q11 >= 3:
        d_items += 1
    q12 = int(request.form['q12'])
    if q12 >= 3:
        d_items += 1
    q13 = int(request.form['q13'])
    if q13 >= 3:
        c_items += 1
    q14 = int(request.form['q14'])
    if q14 >= 3:
        c_items += 1
    q15 = int(request.form['q15'])
    if q15 >= 3:
        d_items += 1
    q16 = int(request.form['q16'])
    if q16 >= 3:
        d_items += 1
    q17 = int(request.form['q17'])
    if q17 >= 3:
        d_items += 1
    q18 = int(request.form['q18'])
    if q18 >= 3:
        d_items += 1
    q19 = int(request.form['q19'])
    if q19 >= 3:
        d_items += 1
    q20 = int(request.form['q20'])
    if q20 >= 3:
        d_items += 1
    total_score = q1+q2+q3+q4+q5+q6+q7+q8+q9+q10+q11+q12+q13+q14+q15+q16+q17+q18+q19+q20
    return render_template('PTSDTest.html', score=total_score,b=b_items,c=c_items,d=d_items)

@app.route('/ADHDTest',methods=['GET', 'POST'])
def ADHDTest():
    return render_template('ADHDTest.html')

@app.route('/calculate3', methods=['GET', 'POST'])
def calculate3():
    q1 = int(request.form['q1'])
    q2 = int(request.form['q2'])
    q3 = int(request.form['q3'])
    q4 = int(request.form['q4'])
    q5 = int(request.form['q5'])
    q6 = int(request.form['q6'])
    q7 = int(request.form['q7'])
    q8 = int(request.form['q8'])
    q9 = int(request.form['q9'])
    q10 = int(request.form['q10'])
    q11 = int(request.form['q11'])
    q12 = int(request.form['q12'])
    q13 = int(request.form['q13'])
    q14 = int(request.form['q14'])
    q15 = int(request.form['q15'])
    q16 = int(request.form['q16'])
    q17 = int(request.form['q17'])
    q18 = int(request.form['q18'])
    q19 = int(request.form['q19'])
    q20 = int(request.form['q20'])
    q21 = int(request.form['q21'])
    q22 = int(request.form['q22'])
    q23 = int(request.form['q23'])
    q24 = int(request.form['q24'])
    q25 = int(request.form['q25'])
    q26 = int(request.form['q26'])
    q27 = int(request.form['q27'])
    q28 = int(request.form['q28'])
    q29 = int(request.form['q29'])
    q30 = int(request.form['q30'])
    q31 = int(request.form['q31'])
    q32 = int(request.form['q32'])
    q33 = int(request.form['q33'])
    q34 = int(request.form['q34'])
    q35 = int(request.form['q35'])
    total_score = q1+q2+q3+q4+q5+q6+q7+q8+q9+q10+q11+q12+q13+q14+q15+q16+q17+q18+q19+q20+q21+q22+q23+q24+q25+q26+q27+q28+q29+q30+q31+q32+q33+q34+q35
    mean = 35 * 1.5  # Assuming mean score of 1.5 per question
    standard_deviation = 0.5  # Assuming standard deviation of 0.5
    raw_score = total_score  # Raw score is the sum of individual question scores
    t_score = 10 * ((raw_score - mean) / standard_deviation)
    return render_template('ADHDTest.html', score=t_score)

@app.route('/predict1',methods=['GET', 'POST'])
def predict1():
    return render_template('ADHDAdvTest.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/clinics',methods=['GET', 'POST'])
def clinics():
    location = request.form.get('location')
    print(location)
    list1 = scrape_clinic_and_rating(location)
    list2 = scrape_clinic_and_rating1(location)
    combined_list = list({clinic['clinic']: clinic for clinic in list1 + list2}.values())
    for clinic in combined_list:
        clinic['Contact and address'] = clinic.get('Contact and address', '').replace('·', ',')
    for clinic in combined_list:
        contact_address = clinic.get('Contact and address', '')
        if ',' in contact_address:
            address, contact = map(str.strip, contact_address.rsplit(',', 1))
            clinic['Address'] = address
            clinic['Contact'] = contact
    print(combined_list)
    return render_template('clinics.html',combined_list=combined_list)

@app.route('/blogs',methods=['GET', 'POST'])
def blogs():
    list1 = scrape_and_run1()
    list2 = scrape_and_run2()
    zipped_list = zip(list1, list2)
    return render_template('blogs.html',combined_list=zipped_list)

@app.route('/therapy',methods=['GET', 'POST'])
def therapy():
    list=scrape_and_run3()
    list1=scrape_and_run4()
    print(list1)
    return render_template('thep.html',list=list,list1=list1)

if __name__ == '__main__':
    app.run(debug=True,port=8080,use_reloader=False)
