import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score
from sklearn.ensemble import IsolationForest

def main():

    st.sidebar.title("Text classification")
    st.title("Text classification")

    def add_radiobutton():

        options = ('Logistic Regression','Decision Tree Classifier', 'K-Nearest Neighbor','Linear Discriminant Analysis','Gaussian Naive Bayes','Random Forest Classifier')
        my_button = st.sidebar.radio("Select a classifier", range(len(options)), format_func=lambda x: options[x])
        classifier_list = [LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier(), LinearDiscriminantAnalysis(), GaussianNB(), RandomForestClassifier()]

        classifier = classifier_list[my_button]
        model_name = options[my_button]

        return classifier, model_name

    @st.cache(persist=True)
    def preprocess_data():

        features_df = pd.read_csv("https://raw.githubusercontent.com/gdeb2/CourseProject/main/Project/features.csv")
        feature_names = ['rating', 'capital_ratio', 'digit_ratio', 'punctuation_ratio', 'word_count', 'character_count', 'capital_letters_count', 'digit_count', 'punctuation_count' , 'sentiment_score']
        X = features_df[feature_names]
        y = features_df['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def load_up_classifier(model_name, classifier, x_train, x_test, y_train, y_test):

        st.header('Classify genuine and fake reviews')
        st.subheader(model_name)
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
        st.subheader("AUC-ROC Curve")
        st.pyplot(fig)

    def bar_plot(s):

        st.write("---------------------------------- \n")
        st.text(s)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        s.plot(kind='bar', ax=ax)
        st.pyplot(fig)

    @st.cache(persist=True)
    def split_data():

        X = pd.read_csv("https://raw.githubusercontent.com/gdeb2/CourseProject/main/Project/cleanReviews.csv")
        label = LabelEncoder()
        for col in X.columns:
            X[col] = label.fit_transform(X[col])
        X_train, X_test = train_test_split(X, random_state=0)
        return X_train, X_test

    def anomaly_detection():

        #Split train and test data for anomaly detection
        X_train, X_test = split_data()
        #Fit and predict model
        model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
        model = model.fit(X_train)
        anomaly_prediction = model.predict(X_test)
        st.header("Anomaly detection")

        X_outliers = anomaly_prediction[anomaly_prediction == -1]
        X_valid = anomaly_prediction[anomaly_prediction == 1]

        st.write("Number of outliers:", len(X_outliers))
        st.write("Number of inliners", len(X_valid))
        st.write("Accuracy:", round(len(X_valid) / len(anomaly_prediction) ,2))
        st.text("(Note: Count based on Training split 75% and Test split 25%)")

    def exploratory_data_analysis():

        st.header('Exploratory Data Analysis')

        path = "https://raw.githubusercontent.com/gdeb2/CourseProject/main/Project/cleanReviews.csv"
        df = pd.read_csv(path)

        lb = "---------------------------------- \n"
        st.write("\n First 5 rows of the dataset",df.head())
        st.write(lb, "Descriptive statistics", df.describe(include = "all"))

        st.write(lb, "Dataset info")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.write(lb, "Number of null values in each column ")
        st.text(df.isna().sum())
        st.write(lb, "Unique classifier")
        st.text(df.label.unique())
        st.write(lb,"Size of unique classifier")
        st.text(df.label.unique().size)
        st.write(lb,"Group dataset by column 'label'")
        st.text(df.groupby(['label']).count())
        st.write(lb,"Number of unique ratings")
        st.text(df.rating.unique())
        st.write(lb, "Group dataset by column 'rating'")
        st.text(df.groupby(['rating']).count())
        st.write(lb, "Number of unique categories")
        st.text(df.category.unique())
        st.write(lb, "Group dataset by column 'category'")
        st.text(df.groupby(['category']).count())

        dataset_copy = df.copy()
        st.write(lb, "Dataset overview",dataset_copy.describe())

        dataset_copy = df.drop(columns=['rating','original_text','clean_text','word_count','character_count','capital_letters_count','digit_count','punctuation_count','sentiment_score'])
        bar_plot(s = dataset_copy.groupby('label').count())
        st.write("Bar chart showing data grouped by column 'label'")
        dataset_copy = df.drop(columns=['category','original_text','clean_text','word_count','character_count','capital_letters_count','digit_count','punctuation_count','sentiment_score'])
        bar_plot(s = dataset_copy.groupby('rating').count())
        st.write("Bar chart showing data grouped by column 'rating'")
        dataset_copy = df.drop(columns=['original_text','clean_text','word_count','character_count','capital_letters_count','digit_count','punctuation_count','sentiment_score'])
        bar_plot(dataset_copy.groupby(['rating','label']).count())
        st.write("Bar chart showing data grouped by columnns 'rating' and 'label'")
        dataset_copy = df.drop(columns=['original_text','clean_text','word_count','character_count','capital_letters_count','digit_count','punctuation_count','sentiment_score'])
        bar_plot(dataset_copy.groupby(['category','label']).count())
        st.write("Bar chart showing data grouped by columns 'category' and 'label'")

        st.write(lb, "Pairwise correlation of columns",df.corr())

    #Navigation menu
    classifier, model_name = add_radiobutton()
    #Split train and test data
    X_train, X_test, y_train, y_test = preprocess_data()

    if st.sidebar.button('Classify'):
        #Exploratory data analysis
        exploratory_data_analysis()
        #Model analysis
        train_model, probs = load_up_classifier(model_name, classifier, X_train, X_test, y_train, y_test)
        #Confusion matrix
        confusion_matrixes(model = train_model, model_name = model_name)
        #Generate AUC (Area Under The Curve)- ROC (Receiver Operating Characteristics) curve
        generate_all_curves(model = train_model, probs = probs)
        #Anomaly detection
        anomaly_detection()

if __name__ == '__main__':
    main()






