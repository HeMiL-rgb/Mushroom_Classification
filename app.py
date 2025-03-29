# import streamlit as st
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import precision_score, recall_score, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier

# def main():
#     st.title("Binary Classification WebApp")    
#     st.markdown("Are your mushrooms edible or poisonous? 🍄")

#     st.sidebar.title("Binary Classification")
#     st.sidebar.markdown("Choose a model and tune hyperparameters:")

#     @st.cache_data
#     def load_data():
#         data = pd.read_csv('mushrooms.csv')
#         label = LabelEncoder()
#         for col in data.columns:
#             data[col] = label.fit_transform(data[col])
#         return data

#     @st.cache_data
#     def split(df):
#         y = df['type']
#         X = df.drop(columns=['type'])
#         return train_test_split(X, y, test_size=0.3, random_state=0)

#     def plot_metrics(metrics_list, model, X_test, y_test, y_pred):
#         if 'Confusion Matrix' in metrics_list:
#             st.subheader("Confusion Matrix")
#             disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
#             st.pyplot(disp.figure_)

#         if 'ROC Curve' in metrics_list:
#             st.subheader("ROC Curve")
#             RocCurveDisplay.from_estimator(model, X_test, y_test).plot()
#             st.pyplot()

#         if 'Precision-Recall Curve' in metrics_list:
#             st.subheader("Precision-Recall Curve")
#             PrecisionRecallDisplay.from_estimator(model, X_test, y_test).plot()
#             st.pyplot()

#     df = load_data()
#     X_train, X_test, y_train, y_test = split(df)
#     class_names = ['Edible', 'Poisonous']

#     st.sidebar.subheader("Choose Classifier")
#     classifier = st.sidebar.selectbox("Classifier", (
#         "Support Vector Machine (SVM)", "Logistic Regression", "Random Forest",
#         "Decision Tree", "K-Nearest Neighbors (KNN)", "Gradient Boosting", "XGBoost"))

#     def train_and_evaluate(model):
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         st.subheader("Model Performance")
#         st.write(f"Precision: {precision_score(y_test, y_pred, average='binary'):.4f}")
#         st.write(f"Recall: {recall_score(y_test, y_pred, average='binary'):.4f}")
        
#         metrics_list = st.sidebar.multiselect("Choose metrics to plot", 
#                                               ['Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'])
#         plot_metrics(metrics_list, model, X_test, y_test, y_pred)

#     if classifier == "Support Vector Machine (SVM)":
#         C = st.sidebar.slider("C (Regularization parameter)", 0.01, 10.0, step=0.01)
#         kernel = st.sidebar.radio("Kernel", ("rbf", "linear"))
#         gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"))
    
#         if st.sidebar.button("Classify"):
#             train_and_evaluate(SVC(C=C, kernel=kernel, gamma=gamma, random_state=0))
    
#     elif classifier == "Logistic Regression":
#         C = st.sidebar.slider("C (Regularization parameter)", 0.01, 10.0, step=0.01)
#         max_iter = st.sidebar.slider("Max Iterations", 100, 500)
    
#         if st.sidebar.button("Classify"):
#             train_and_evaluate(LogisticRegression(C=C, max_iter=max_iter, random_state=0))
    
#     elif classifier == "Random Forest":
#         n_estimators = st.sidebar.slider("Number of Trees", 100, 500, step=10)
#         max_depth = st.sidebar.slider("Max Depth", 1, 20)
    
#         if st.sidebar.button("Classify"):
#             train_and_evaluate(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0))

#     elif classifier == "Decision Tree":
#         max_depth = st.sidebar.slider("Max Depth", 1, 20)
    
#         if st.sidebar.button("Classify"):
#             train_and_evaluate(DecisionTreeClassifier(max_depth=max_depth, random_state=0))

#     elif classifier == "K-Nearest Neighbors (KNN)":
#         n_neighbors = st.sidebar.slider("Number of Neighbors (K)", 1, 20)
    
#         if st.sidebar.button("Classify"):
#             train_and_evaluate(KNeighborsClassifier(n_neighbors=n_neighbors))

#     elif classifier == "Gradient Boosting":
#         learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, step=0.01)
#         n_estimators = st.sidebar.slider("Number of Estimators", 100, 500, step=10)
    
#         if st.sidebar.button("Classify"):
#             train_and_evaluate(GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=0))

#     elif classifier == "XGBoost":
#         learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, step=0.01)
#         n_estimators = st.sidebar.slider("Number of Trees", 100, 500, step=10)
    
#         if st.sidebar.button("Classify"):
#             train_and_evaluate(XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=0))

#     if st.sidebar.checkbox("Show raw data", False):
#         st.subheader("Mushroom Data Set (Classification)")
#         st.write(df)

# if __name__ == '__main__':
#     main()

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def main():
    st.title("Binary Classification WebApp")    
    st.markdown("Are your mushrooms edible or poisonous? 🍄")

    @st.cache_data
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    df = load_data()
    X = df.drop(columns=['type'])
    y = df['type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    class_names = ['Edible', 'Poisonous']

    st.subheader("Enter Mushroom Features")
    user_input = []
    for col in X.columns:
        value = st.selectbox(f"{col}", sorted(df[col].unique()))
        user_input.append(value)
    
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", (
        "Support Vector Machine (SVM)", "Logistic Regression", "Random Forest",
        "Decision Tree", "K-Nearest Neighbors (KNN)", "Gradient Boosting", "XGBoost"))

    def train_and_evaluate(model, input_data):
        model.fit(X_train, y_train)
        prediction = model.predict([input_data])[0]
        st.subheader("Prediction")
        st.write("The mushroom is:", "Poisonous" if prediction else "Edible")
    
    if classifier == "Support Vector Machine (SVM)":
        C = st.sidebar.slider("C (Regularization parameter)", 0.01, 10.0, step=0.01)
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"))
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"))
    
        if st.button("Classify"):
            train_and_evaluate(SVC(C=C, kernel=kernel, gamma=gamma, random_state=0), user_input)
    
    elif classifier == "Logistic Regression":
        C = st.sidebar.slider("C (Regularization parameter)", 0.01, 10.0, step=0.01)
        max_iter = st.sidebar.slider("Max Iterations", 100, 500)
    
        if st.button("Classify"):
            train_and_evaluate(LogisticRegression(C=C, max_iter=max_iter, random_state=0), user_input)
    
    elif classifier == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Trees", 100, 500, step=10)
        max_depth = st.sidebar.slider("Max Depth", 1, 20)
    
        if st.button("Classify"):
            train_and_evaluate(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0), user_input)

    elif classifier == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 1, 20)
    
        if st.button("Classify"):
            train_and_evaluate(DecisionTreeClassifier(max_depth=max_depth, random_state=0), user_input)

    elif classifier == "K-Nearest Neighbors (KNN)":
        n_neighbors = st.sidebar.slider("Number of Neighbors (K)", 1, 20)
    
        if st.button("Classify"):
            train_and_evaluate(KNeighborsClassifier(n_neighbors=n_neighbors), user_input)

    elif classifier == "Gradient Boosting":
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, step=0.01)
        n_estimators = st.sidebar.slider("Number of Estimators", 100, 500, step=10)
    
        if st.button("Classify"):
            train_and_evaluate(GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=0), user_input)

    elif classifier == "XGBoost":
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, step=0.01)
        n_estimators = st.sidebar.slider("Number of Trees", 100, 500, step=10)
    
        if st.button("Classify"):
            train_and_evaluate(XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=0), user_input)

if __name__ == '__main__':
    main()
