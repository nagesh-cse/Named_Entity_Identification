import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys
import argparse
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle as pkl
import streamlit as st
import numpy as np
import nltk
from preprocess_csv import preprocess_data, generate_features_ner
import argparse
import matplotlib.pyplot as plt



def main(train_file, test_file, ngram, kernel='rbf'):
    X_train, X_test, y_train, y_test = preprocess_data(train_file, test_file, ngram)

    # Train SVM Classifier
    svm_clf = SVC(kernel=kernel)
    svm_clf.fit(X_train, y_train)

    with open(f'svm_ner_classifier_ngram_{str(ngram)}_{kernel}_char.pkl', 'wb') as file:
        pkl.dump(svm_clf, file)

    # Predict NER label
    y_pred = svm_clf.predict(X_test)
    # Print classification report
    print(classification_report(y_test, y_pred))
    
    with open(f'classification_report_ngram_{str(ngram)}_{kernel}_char.txt', 'w') as file:
        file.write(str(classification_report(y_test, y_pred)))

    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
                'No-Named', 'Named'], yticklabels=['No-Named', 'Named'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confMat_{str(ngram)}_{kernel}.png', dpi=600)



def predict_pretrained(svm_clf_file, corpus, ngram):
    with open(svm_clf_file, 'rb') as file:
        svm_clf = pkl.load(file)

    X, words = generate_features_ner(corpus, ngram)
    nerPred = svm_clf.predict(X)

    nerTagged_words = []
    for i in range(len(words)):
        nerTagged_words.append(words[i] + '_' + str(nerPred[i]))

    nerTagged_words = ' '.join(nerTagged_words)

    return nerTagged_words




if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("ERROR : Providde Kernal name and N(widow for POS tags)")
        sys.exit()

    train_file = 'squad-train.csv'
    test_file = 'squad-test.csv'
    kernel = sys.argv[1]

    try:
        ngram = int(sys.argv[2])
    except:
        print("Provide N as integer")
        sys.exit()
    
    if sys.argv[3] == 'train':
        main(train_file, test_file, ngram, kernel)
    elif sys.argv[3] != 'test':
        print("Provide correct 'Train' or 'Test' argument")
        sys.exit()

    sentence = st.text_input("Enter a sentence")
    if sentence:
        nerTagged_words = predict_pretrained(f'svm_ner_classifier_ngram_{str(ngram)}_{kernel}_char.pkl', sentence, ngram)
        print(nerTagged_words)
        st.write("Output:\n\n " + nerTagged_words)