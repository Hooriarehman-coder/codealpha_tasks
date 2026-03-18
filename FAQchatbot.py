import tkinter as tk
from tkinter import scrolledtext
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')


faq_questions = [
    "What is your return policy?",
    "How can I track my order?",
    "Do you offer cash on delivery?",
    "How long does shipping take?",
    "How can I contact customer support?"
]

faq_answers = [
    "You can return the product within 7 days of delivery.",
    "You can track your order using the tracking link sent to your email.",
    "Yes, we offer cash on delivery service.",
    "Shipping usually takes 3-5 business days.",
    "You can contact customer support at support@example.com."
]




def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    return " ".join(tokens)


# Preprocess FAQ questions
processed_questions = [preprocess(q) for q in faq_questions]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_questions)



def get_response(user_input):
    user_input_processed = preprocess(user_input)
    user_vector = vectorizer.transform([user_input_processed])

    similarity = cosine_similarity(user_vector, tfidf_matrix)
    index = similarity.argmax()

    if similarity[0][index] < 0.3:
        return "Sorry, I don't understand your question."
    else:
        return faq_answers[index]



def send_message():
    user_input = entry.get()
    if user_input.strip() == "":
        return

    chat_area.insert(tk.END, "You: " + user_input + "\n")
    response = get_response(user_input)
    chat_area.insert(tk.END, "Bot: " + response + "\n\n")

    entry.delete(0, tk.END)


# Create Window
window = tk.Tk()
window.title("FAQ Chatbot")
window.geometry("500x500")

chat_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=60, height=20)
chat_area.pack(pady=10)

entry = tk.Entry(window, width=40)
entry.pack(pady=5)

send_button = tk.Button(window, text="Send", command=send_message)
send_button.pack()

window.mainloop()