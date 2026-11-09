import os,joblib,pandas as pd,seaborn as sns,matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import *
os.makedirs("models",exist_ok=True);os.makedirs("artifacts",exist_ok=True)
df=pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
y=df['Attrition'].map({'No':0,'Yes':1});X=df.drop(columns=['Attrition'])
cat=X.select_dtypes(include='object').columns;num=X.select_dtypes(exclude='object').columns
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
Xtrp=pre.fit_transform(Xtr);Xtep=pre.transform(Xte)
joblib.dump(pre,'models/preprocessor.pkl')
models={'logistic_regression':LogisticRegression(max_iter=2000),'decision_tree':DecisionTreeClassifier(max_depth=6,random_state=42),'knn':KNeighborsClassifier(7),'naive_bayes':GaussianNB(),'svm':SVC(probability=True)}
rows=[]
for n,m in models.items():
 a=Xtrp.toarray() if n=='naive_bayes' and hasattr(Xtrp,'toarray') else Xtrp
 b=Xtep.toarray() if n=='naive_bayes' and hasattr(Xtep,'toarray') else Xtep
 m.fit(a,ytr);p=m.predict(b);pr=m.predict_proba(b)[:,1]
 rows.append([n,accuracy_score(yte,p),roc_auc_score(yte,pr),precision_score(yte,p,zero_division=0),recall_score(yte,p,zero_division=0),f1_score(yte,p,zero_division=0),matthews_corrcoef(yte,p)])
 cm=confusion_matrix(yte,p)
 plt.figure();sns.heatmap(cm,annot=True,fmt='d');plt.savefig(f'artifacts/{n}_cm.png');plt.close()
 joblib.dump(m,f'models/{n}.pkl')
pd.DataFrame(rows,columns=['Model','Accuracy','AUC','Precision','Recall','F1','MCC']).to_csv('models/metrics.csv',index=False)
xt=Xte.copy();xt['Attrition']=yte.map({0:'No',1:'Yes'});xt.to_csv('test_data/employee_test_data.csv',index=False)
print('Training complete')