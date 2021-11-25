import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score

def main():

    st.sidebar.title("Text classification")
    st.title("Classify genuine and fake reviews")

    def add_radiobutton():

        options = ('Logistic Regression','Decision Tree Classifier', 'K-Nearest Neighbor','Linear Discriminant Analysis','Gaussian Naive Bayes','Random Forest Classifier')
        my_button = st.sidebar.radio("Select a classifier", range(len(options)), format_func=lambda x: options[x])
        classifier_list = [LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier(), LinearDiscriminantAnalysis(), GaussianNB(), RandomForestClassifier()]

        classifier = classifier_list[my_button]
        model_name = options[my_button]
        st.subheader(model_name)

        return classifier, model_name

    def preprocess_data():

        features_df = pd.read_csv("features.csv")
        feature_names = ['rating', 'capital_ratio', 'digit_ratio', 'punctuation_ratio', 'word_count', 'character_count', 'capital_letters_count', 'digit_count', 'punctuation_count' , 'sentiment_score']
        X = features_df[feature_names]
        y = features_df['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def load_up_classifier(classifier, x_train, x_test, y_train, y_test):

        model = classifier
        model = model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        probs = model.predict_proba(x_test)

        st.write("Accuracy ", model.score(X_test, y_test).round(2))
        st.write("Precision: ", precision_score(y_test, prediction).round(2))
        st.write("Recall: ", recall_score(y_test, prediction).round(2))

        return model, probs


    def confusion_matrixes(model, model_name):

        fig, ax = plt.subplots(nrows=1, ncols=1)
        plot_confusion_matrix(model,
                              X_test,
                              y_test,
                              ax = ax,
                              cmap='YlGnBu',
                              display_labels=["True","Fake"])
        ax.title.set_text(model_name)
        st.subheader("Confusion Matrix")
        st.pyplot(fig)

    def generate_all_curves(model, probs):

        preds = probs[:,1]
        fpr, tpr, threshold = roc_curve(y_test, preds)
        roc_auc = auc(fpr, tpr)
        fig, axs = plt.subplots(1, 1) #subplot_kw=dict(polar=True))
        axs.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        axs.legend(loc = 'lower right')
        axs.plot([0, 1], [0, 1],'r--')
        axs.set_title(model)
        axs.set_ylabel('True Positive Rate')
        axs.set_xlabel('False Positive Rate')
        st.subheader("AUC and ROC Curve")
        st.pyplot(fig)

    classifier, model_name = add_radiobutton()
    X_train, X_test, y_train, y_test = preprocess_data()

    if st.sidebar.button('Classify'):
        train_model, probs = load_up_classifier(classifier, X_train, X_test, y_train, y_test)
        confusion_matrixes(model = train_model, model_name = model_name)
        #Generate AUC (Area Under The Curve)- ROC (Receiver Operating Characteristics) curve
        generate_all_curves(model = train_model, probs = probs)

if __name__ == '__main__':
    main()






