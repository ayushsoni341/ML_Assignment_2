import streamlit as st,pandas as pd,joblib,plotly.express as px
from sklearn.metrics import *
st.set_page_config(page_title='IBM HR Attrition',layout='wide')
st.title('IBM HR Analytics Employee Attrition Dashboard')
try:
 mdf=pd.read_csv('models/metrics.csv')
 st.subheader('Model Comparison')
 st.dataframe(mdf,use_container_width=True)
 st.plotly_chart(px.bar(mdf,x='Model',y='Accuracy'),use_container_width=True)
except: st.info('Run train_models.py first')
file=st.file_uploader('Upload employee_test_data.csv',type=['csv'])
model=st.selectbox('Choose Model',['logistic_regression','decision_tree','knn','naive_bayes','svm'])
if file:
 df=pd.read_csv(file);y=df['Attrition'].map({'No':0,'Yes':1});X=df.drop(columns=['Attrition'])
 pre=joblib.load('models/preprocessor.pkl');mdl=joblib.load(f'models/{model}.pkl')
 Xp=pre.transform(X)
 if model=='naive_bayes' and hasattr(Xp,'toarray'):Xp=Xp.toarray()
 p=mdl.predict(Xp);pr=mdl.predict_proba(Xp)[:,1]
 c1,c2,c3=st.columns(3)
 c1.metric('Accuracy',round(accuracy_score(y,p),4));c2.metric('AUC',round(roc_auc_score(y,pr),4));c3.metric('F1',round(f1_score(y,p),4))
 st.text(classification_report(y,p))