import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 


st.set_page_config("Dataset Analysis")
st.sidebar.markdown(""" # **Step 1: Upload File**""")
dt=st.sidebar.file_uploader(label="",type="CSV")
st.sidebar.markdown(""" # **Step 2: Select One**""")
option=st.sidebar.radio(label="",options=["Exploratory Data Analysis","Plotting"])

if option=="Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")

    if dt:
        df=pd.read_csv(dt)
        
        with st.beta_expander("Show Dataset"):
            st.write(df)

        with st.beta_expander("Columns with data type "):
            st.write(df.dtypes)

        with st.beta_expander("Shape"):
            st.write(f"Dataset Contains **{len(df.index)}** Rows and **{len(df.columns)}** Columns" )

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
    st.title("Dataset Plotting")

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
            sns.heatmap(df.corr(),annot=True)
            plt.yticks(rotation = 90)
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
