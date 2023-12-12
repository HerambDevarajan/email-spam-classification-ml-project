import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    word_list=[]

    for i in text:
        if i.isalnum():
            word_list.append(i)

    text = word_list[:]
    word_list.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            word_list.append(i)

    text = word_list[:]
    word_list.clear()

    for i in text:
        word_list.append(ps.stem(i))

    return " ".join(word_list)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model1.pkl','rb'))

def main():
    st.title("Email Spam Classifier")

    # Heading text area
    heading_text = st.text_area("Enter your email body here:", "")
    

    # Predict button
    if st.button("Predict"):
        transformed_email = transform_text(heading_text)
        vector_input = tfidf.transform([transformed_email]).toarray()
        result = model.predict(vector_input)[0]
        outtext = "Spam" if result == 1 else "Not Spam"
        probabilities = model.predict_proba(vector_input)[0]
        probability_spam = probabilities[1]  # Probability of being spam

        # Display the prediction result
        st.subheader("Prediction Result:")
        st.success(f"The entered text is: **{outtext}**")

        # Display the probability of being spam
        st.success(f"Probability of Being Spam: {probability_spam * 100:.2f}%")

if __name__ == "__main__":
    main()
