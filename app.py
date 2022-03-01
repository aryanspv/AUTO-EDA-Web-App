#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix, classification_report
from sklearn.decomposition import PCA

st.set_page_config("Dataset Analysis")
st.title("AUTO EDA WEB APP")
st.sidebar.markdown(""" # **Step 1: Upload File**""")
dt=st.sidebar.file_uploader(label="",type="CSV")
st.sidebar.markdown(""" # **Step 2: Select One**""")
option=st.sidebar.radio(label="",options=["Exploratory Data Analysis","Plotting","Machine Learning"])

if option=="Exploratory Data Analysis":
    st.subheader("Exploratory Data Analysis")

    if dt:
        df=pd.read_csv(dt)
        
        with st.beta_expander("Show Dataset"):
            st.write(df)

        with st.beta_expander("Columns with data type "):
            st.write(df.dtypes)

        with st.beta_expander("Shape"):
            st.write(f"Rows:**{len(df.index)}**" )
            st.write(f"Columns:**{len(df.columns)}**")

        with st.beta_expander("Summary"):
            st.write(df.describe())

        with st.beta_expander("Null Values"):
            st.write(df.isnull().sum())

        with st.beta_expander("Select Multiple Columns"):
            c=st.multiselect("",df.columns)
            st.write(df[c])

        with st.beta_expander("Value Count"):
            v=st.selectbox("",df.columns)
            st.write(df[v].value_counts())

        with st.beta_expander("Correlation Chart"):
            st.write(df.corr())
    else:
        st.warning("Please Upload a CSV file")


if option=="Plotting":
    st.subheader("Dataset Plotting")

    if dt:
        df=pd.read_csv(dt)
        df.dropna(inplace=True)
        st.markdown("""We have removed Null Values and divided the columns in following 2 categories.

        1. Categorical Columns
        2. Numerical Columns
        """)
        df_num=df.select_dtypes(include=[np.float64,np.int64])
        df_category=df.select_dtypes(include="object")
        plot_list=["Distribution Plot","Jointplot","Count Plot","Pair Plot","Bar Plot",
                    "Box Plot","Correlation Plot","Scatter Plot","Area Chart"]
        slct_plot=st.selectbox("Select Plot type :",plot_list)
        
        if slct_plot=="Distribution Plot":
            s=st.selectbox("Select Column (Numerical Columns)",df_num.columns)
            fig,ax=plt.subplots()
            sns.distplot(df[s])
            st.pyplot(fig=fig)

        if slct_plot=="Jointplot":
            first,second=st.beta_columns(2)
            a=first.selectbox("Select Column",df_num.columns)
            b=second.selectbox("Select Columns",df_num.columns)
            fig,ax=plt.subplots()
            fig=sns.jointplot(x=df[a],y=df[b],data=df,kind="hex")
            st.pyplot(fig=fig)

        if slct_plot=="Bar Plot":
            first,second=st.beta_columns(2)
            a=first.selectbox("Select Categorical Column",df_category.columns)
            b=second.selectbox("Select Numerical value Columns",df_num.columns)
            fig,ax=plt.subplots()
            sns.barplot(x=df[a],y=df[b],data=df)
            plt.xticks(rotation=75)
            st.pyplot(fig)

        if slct_plot=="Count Plot":
            s=st.selectbox("Select Categorical Column",df_category.columns)
            fig,ax=plt.subplots()
            sns.countplot(x=df[s],data=df)
            plt.xticks(rotation=75)
            st.pyplot(fig=fig)

        if slct_plot=="Pair Plot":
            pp=sns.pairplot(df)
            st.pyplot(pp)

        if slct_plot=="Box Plot":
            first,second=st.beta_columns(2)
            a=first.selectbox("Select Categorical Column",df_category.columns)
            b=second.selectbox("Select Numerical value Columns",df_num.columns)
            fig,ax=plt.subplots()
            sns.boxplot(x=df[a],y=df[b],data=df)
            plt.xticks(rotation=75)
            st.pyplot(fig)

        if slct_plot=="Correlation Plot":
            fig,ax=plt.subplots()
            sns.heatmap(df.corr(),annot=False,fmt="d",robust=True,cmap='coolwarm')
            st.pyplot(fig)

        if slct_plot=="Scatter Plot":
            first,second=st.beta_columns(2)
            a=first.selectbox("Select Categorical Column",df_category.columns)
            b=second.selectbox("Select Numerical value Columns",df_num.columns)
            fig,ax=plt.subplots()
            sns.scatterplot(x=df[a],y=df[b],data=df)
            plt.xticks(rotation=75)
            st.pyplot(fig)

        if slct_plot=="Area Chart":
            a=st.selectbox("Select Numerical value Columns",df_num.columns)
            fig, ax = plt.subplots()
            df[a].plot(kind="area")
            st.pyplot(fig)

    else:
        st.warning("Please Upload a CSV file")


if option=="Machine Learning":

    st.subheader("Machine Learning")
    st.markdown("""**Machine Learning is implemented on Scikit-Learn Built in Data Sets for better understanding**""")
    dataset_name=st.selectbox("Select DataSet",("Iris","Breast Cancer","Wine"))
    classifier_name=st.selectbox("Select Classifier",("KNN","SVM","Random Forest"))

    def get_dataset(dataset_name):
    
        if dataset_name=="Iris":
            data = datasets.load_iris()

        elif dataset_name=="Breast Cancer":
            data = datasets.load_breast_cancer()

        else:
            data = datasets.load_wine()
    
        X=data.data
        y=data.target
        return X,y

    X,y=get_dataset(dataset_name)
    st.write("Shape of dataset :",X.shape)
    
    def add_parameter_ui(clf_name):
        params=dict()
       
        if clf_name=="KNN":
            K = st.slider("K",1,15,step=1)
            params["K"]=K
            train_split=st.slider("Train Test Split",1,99,30,1)
            params["train_split"]=train_split

        elif clf_name=="SVM":
            C = st.slider("C",0.01,10.0)
            params["C"]=C
            train_split=st.slider("Train Test Split",1,99,30,1)
            params["train_split"]=train_split

        else:
            max_depth = st.slider("max depth",2,15)
            n_estimators = st.slider("n_estimator",1,100)
            params["max_depth"] = max_depth
            params["n_estimators"] = n_estimators
            train_split=st.slider("Train Test Split",1,99,30,1)
            params["train_split"]=train_split

        return params

    params = add_parameter_ui(classifier_name)
    
    def get_classifier(clf_name,params):

        if clf_name=="KNN":
            clf = KNeighborsClassifier(n_neighbors=params["K"])

        elif clf_name=="SVM":
            clf = SVC(C=params["C"])

        else:
            clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"],random_state=1234)
    
        return clf

    clf=get_classifier(classifier_name,params)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234,stratify=y)
    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)

    acc=accuracy_score(y_test,y_pred)
    st.markdown(f"""Accuracy : ** {acc*100} **% """)
    cl=classification_report(y_test,y_pred)
    st.markdown(f""" **Classification Report : ** {cl} ** """)

## Ploting
    pca = PCA(2)
    X_projeted = pca.fit_transform(X)
    X1=X_projeted[:,0]
    X2=X_projeted[:,1]
    fig,ax=plt.subplots()
    plt.scatter(X1,X2,c=y,alpha=0.8,cmap="viridis")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    st.pyplot(fig)

else:
    pass









