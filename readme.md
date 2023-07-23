# CIxD Aipowered Studying System

Demo: https://cixd-aipowered-studying-system.streamlit.app

## Introduction
------------
CIxD AIpowered Studying System is a Python application that allows you to chat PDF documents, research papers to be specific. The application offers 4 modes; summarization, question generation, answer generation and topic recommendation. This app provides the option to choose a diverse range of language models to generate answers to your queries. Please note that the app will only respond to questions related to the loaded PDFs.

## How It Works
------------

## Dependencies and Installation
----------------------------
To install 'CIxD Aipowered Studying System', please follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

3. Obtain an API key from OpenAI, Replicate or HuggingFace nd add it to the `.env` file in the project directory.

## Usage
-----
To use the MultiPDF Chat App, follow these steps:

1. Run the `app.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

2. The application will launch in your default web browser, displaying the user interface.

3. Load your own PDF documents into the app or select from the provided list.

4. Select your preferred mode and interact.
