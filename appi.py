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
model = pickle.load(open('model.pkl','rb'))

def main():
    st.title("Email Spam Classifier")

    # Heading text area
    heading_text = st.text_area("Enter your email body here:", "")
    

    # Predict button
    if st.button("Predict"):
        # Display the entered text below the button
        transformed_email=transform_text(heading_text)
        #print(transformed_email)
        vector_input = tfidf.transform([transformed_email]).toarray()
        result = model.predict(vector_input)[0]
        outtext=""
        if result==1:
            outtext="Spam"
        else:
            outtext="Not Spam"
        probabilities = model.predict_proba(vector_input)[0]
        probability_spam = probabilities[1]  # Probability of being spam
        st.success(f"Entered Text: {outtext}")

if __name__ == "__main__":
    main()
