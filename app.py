# streamlit_app.py

import streamlit as st
import tensorflow as tf
import re
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
print("Import Sucessful")

# Load BERT tokenizer and model

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=9)

restored_model = tf.keras.Sequential([
    tf.keras.Input(shape=(None,), dtype=tf.int32, name='input_ids'),
    tf.keras.Input(shape=(None,), dtype=tf.int32, name='attention_mask'),
    model,
    tf.keras.layers.Dense(9, activation='softmax')
])
restored_model.load_weights('/content/weights_from_model')

print("Model Loaded Sucessfully")

#Using the same custom funtion and mapping
category_mapping = {'iPhone' :0, 'iPad or iPhone App': 1 , 'iPad': 2 , 'Google': 3, 'Android': 4,
       'Apple': 5 , 'Android App': 6 , 'Other Google product or service': 7,
       'Other Apple product or service': 8 }

reverse_mapping = {v: k for k, v in category_mapping.items()}

def predict_class(single_input_text):
  if type(single_input_text) == str:
    #Applying the same pre-processing to new data
    single_input_text = single_input_text.lower()
    single_input_text = re.sub(r'[^a-z]+',' ',single_input_text)
    single_input_text = re.sub('http[s]?://\S+',' ', single_input_text)
    print(single_input_text)

    #Tokenization, and then input to the model
    single_input_tokens = tokenizer(single_input_text, padding=True, truncation=True, return_tensors='tf')
    output_probabilities = restored_model.predict({'input_ids': single_input_tokens['input_ids'], 'attention_mask': single_input_tokens['attention_mask']})
    predicted_probabilities = np.squeeze(output_probabilities)
    predicted_class = np.argmax(predicted_probabilities)
    category_name = reverse_mapping.get(predicted_class)
    print(f'Predicted Probabilities: {predicted_probabilities}')
    print(f'Predicted Class: {category_name}')
    return category_name
  else:
    print('Input not in String Format')

print('Function defined')

# Streamlit app
st.title("Tweet Product/Service Identifier")

# User input
user_input = st.text_input("Enter / Paste the tweet below:")

if user_input:
    result = predict_class(user_input)
    # Display prediction
    st.write("Predicted Porduct/Service:", result)

# Footer
st.markdown("---")
st.write("This app uses a pre-trained BERT model, created by Sarthak.")