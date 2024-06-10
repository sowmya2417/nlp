import streamlit as st
import pandas as pd
import io

import spacy

# Load a pre-trained English language model
nlp = spacy.load("en_core_web_sm")

def classify_text(text):
    # Process the input text using spaCy
    doc = nlp(text)
    
    # Initialize counters for various linguistic features
    num_nouns = 0
    num_verbs = 0
    num_adjectives = 0
    num_adverbs = 0
    num_pronouns = 0
    num_aux_verbs = 0
    
    # Iterate over tokens in the processed text
    for token in doc:
        # Count nouns
        if token.pos_ == 'NOUN':
            num_nouns += 1
        # Count verbs
        elif token.pos_ == 'VERB':
            num_verbs += 1
        # Count adjectives
        elif token.pos_ == 'ADJ':
            num_adjectives += 1
        # Count adverbs
        elif token.pos_ == 'ADV':
            num_adverbs += 1
        # Count pronouns
        elif token.pos_ == 'PRON':
            num_pronouns += 1
        # Count auxiliary verbs
        elif token.dep_ == 'aux':
            num_aux_verbs += 1
    
    # Determine if the text is more likely to be human-generated or AI-generated
    if num_nouns > 10 and num_verbs > 5 and num_adjectives > 2:
        return 1  # AI-generated
    elif num_pronouns > 5 and num_aux_verbs > 3:
        return 0  # Human-generated
    else:
        return -1  # Uncertain


st.balloons()
st.markdown("# Text Classification App")

st.write("Step into the Text Classification App! 🚀 "
         "Here, we embark on a journey to decipher the origins of text – whether it's crafted by human hands (0) or born from the minds of AI (1). "
         "Choose your path: upload a CSV or text file teeming with words, or plunge into the depths of your thoughts and type directly. The adventure awaits!")

# Text area for typing data
text_input = st.text_area("Enter text here:")

# File upload option
uploaded_file = st.file_uploader("Upload CSV or Text file", type=["csv", "txt"])

if uploaded_file is not None:
    content = uploaded_file.getvalue().decode("utf-8")
    
    # If the file is a CSV, read it with pandas
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(io.StringIO(content))
    # If the file is a text file, create a DataFrame with one column
    elif uploaded_file.type == "text/plain":
        df = pd.DataFrame({"Text": content.split("\n")})
    
    st.write("Here is the data from the uploaded file:")
    st.write(df)
    
    st.write("Classifying text from the uploaded file...")

    # Apply classification function to each text entry
    df['Label'] = df['Text'].apply(classify_text)

    st.write("Here is the classified data:")
    st.write(df)

if text_input:
    st.write("Classifying typed text...")

    # Classify typed text
    text_classification = classify_text(text_input)

    st.write(f"The typed text is classified as {'AI-generated' if text_classification == 1 else 'human-generated'}.")
