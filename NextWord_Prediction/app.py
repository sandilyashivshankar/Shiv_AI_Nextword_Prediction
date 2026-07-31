import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Next Word Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:Poppins;
}

.stApp{
    background: linear-gradient(135deg,#0f172a,#111827,#1e293b);
    color:white;
}

/* Hide Streamlit Footer */
footer{
visibility:hidden;
}

header{
visibility:hidden;
}

/* Sidebar */

section[data-testid="stSidebar"]{
background:#111827;
}

/* Glass Card */

.glass{

background:rgba(255,255,255,.06);

backdrop-filter:blur(15px);

padding:25px;

border-radius:20px;

border:1px solid rgba(255,255,255,.15);

box-shadow:0 10px 40px rgba(0,0,0,.3);

}

/* Button */

div.stButton > button{

background:linear-gradient(90deg,#7c3aed,#2563eb);

color:white;

font-size:18px;

font-weight:600;

border:none;

padding:12px 25px;

border-radius:12px;

transition:.3s;

width:100%;
}

div.stButton > button:hover{

transform:scale(1.03);

background:linear-gradient(90deg,#9333ea,#3b82f6);

}

/* Text Area */

textarea{

font-size:18px !important;

border-radius:12px !important;

}

/* Result Card */

.result{

background:#1E293B;

padding:25px;

border-radius:20px;

border-left:6px solid #7C3AED;

margin-top:20px;

text-align:center;

}

.big{

font-size:35px;

font-weight:bold;

color:#60A5FA;

}

.small{

font-size:18px;

color:#CBD5E1;

}

.title{

text-align:center;

font-size:45px;

font-weight:700;

margin-bottom:0px;

}

.subtitle{

text-align:center;

font-size:18px;

color:#94A3B8;

margin-bottom:30px;

}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lstm_model (1).h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl","rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_maxlen():
    with open("max_len.pkl","rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()
max_len = load_maxlen()

# -----------------------------
# Prediction Function
# -----------------------------

def predict_next_word(text):

    token_list = tokenizer.texts_to_sequences([text])[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_len-1,
        padding="pre"
    )

    predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]

    output = ""

    for word,index in tokenizer.word_index.items():

        if index == predicted:
            output = word
            break

    return output

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.title("🤖 AI Model")

    st.markdown("---")

    st.info("""
Deep Learning based
Next Word Prediction
using

• TensorFlow

• Keras

• LSTM
""")

    st.markdown("---")

    st.metric("Framework","TensorFlow")

    st.metric("Model","LSTM")

    st.metric("Language","Python")

    st.markdown("---")

    st.success("Ready for Prediction")

# -----------------------------
# Main UI
# -----------------------------

st.markdown("<div class='title'>Shiv 🤖 AI Next Word Predictor</div>",unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Deep Learning Powered by TensorFlow & LSTM</div>",unsafe_allow_html=True)

col1,col2=st.columns([2,1])

with col1:

    st.markdown("<div class='glass'>",unsafe_allow_html=True)

    st.subheader("📝 Enter Text")

    text = st.text_area(
        "",
        height=170,
        placeholder="Example:\nArtificial Intelligence is\n\nMachine Learning can\n\nDeep Learning models..."
    )

    predict = st.button("🚀 Predict Next Word")

    st.markdown("</div>",unsafe_allow_html=True)

with col2:

    st.markdown("<div class='glass'>",unsafe_allow_html=True)

    st.subheader("📊 Model Information")

    st.metric("Architecture","LSTM")

    st.metric("Sequence Length",max_len)

    st.metric("Vocabulary",len(tokenizer.word_index))

    st.metric("Status","Ready")

    st.markdown("</div>",unsafe_allow_html=True)

# -----------------------------
# Prediction
# -----------------------------

if predict:

    if text.strip()=="":

        st.warning("Please enter some text.")

    else:

        with st.spinner("Analyzing context..."):

            word = predict_next_word(text)

        st.markdown(f"""

        <div class="result">

        <div class="small">✨ Suggested Next Word</div>

        <div class="big">{word}</div>

        </div>

        """,unsafe_allow_html=True)

# -----------------------------
# Tabs
# -----------------------------

tab1,tab2,tab3 = st.tabs(
    [
        "💡 Examples",
        "⚙️ Workflow",
        "📖 About"
    ]
)

with tab1:

    st.code("""
Machine learning is

Artificial intelligence can

Data Science helps

Deep Learning models

The future of AI
""")

with tab2:

    st.markdown("""

""")

with tab3:

    st.write("""

### About

This project demonstrates an **LSTM-based Next Word Prediction System** trained on sequential text data.

The application predicts the most probable next word based on user input using:

- TensorFlow
- Keras
- LSTM Neural Networks
- Tokenization
- Sequence Padding

Suitable for demonstrating **Natural Language Processing (NLP)** and **Deep Learning** concepts.

""")

# -----------------------------
# Footer
# -----------------------------

st.markdown("---")

st.markdown(
"""
<center>

Developed by <b>Shiv Shankar Tiwari</b>

AI • Deep Learning • NLP • TensorFlow

</center>
""",
unsafe_allow_html=True
)