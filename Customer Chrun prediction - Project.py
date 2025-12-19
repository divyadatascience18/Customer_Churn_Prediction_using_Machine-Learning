import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve
from xgboost import XGBClassifier
import joblib
from tkinter import *
import warnings
warnings.filterwarnings("ignore")
    
    #supervised models used for this project - target label present in the data (CHURN)
print("CUSTOMER CHURN PREDICTION PROJECT")

df=pd.read_csv("C://Users//Divya//Downloads//WA_Fn-UseC_-Telco-Customer-Churn.csv")
    # print(df)
print("-------------------------First 10 rows of the dataset-------------------------")
print("HEAD",df.head(10))
print("-------------------------No.of.rows and columns-------------------------")
print("shape",df.shape)
print("-------------------------Columns details-------------------------")
print("COLUMNS",df.columns)
print("-------------------------Removing duplicates-------------------------")
print("removing duplicates",df.drop_duplicates(inplace=True))
print("-------------------------Target Label details-------------------------")
print("churn details",df['Churn'].value_counts())
Churn_actual=df['Churn'].value_counts()

le=LabelEncoder()

for col in df.columns:
 if df[col].dtype=='object':
  df[col]=le.fit_transform(df[col])
    #split
X=df.drop('Churn',axis=1)
y=df['Churn']



    #train the mode
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,train_size=0.8,random_state=42)
print("-------------------------Test,Train Details-------------------------")
print("X's Test shape",X_test.shape)
print("y's Train shape",X_train.shape)
print("X's Test shape",y_test.shape)
print("y's Train shape",y_train.shape)


    #LOGISTIC REGRESSION
    
logistic_reg_model=LogisticRegression(max_iter=100)
logistic_reg_model.fit(X_train,y_train)

    # joblib.dump(logistic_reg_model,"logistic_chrun_model.pk1")
    # print("model saved")

    #PREDICT

y_pred=logistic_reg_model.predict(X_test)


    #print("predicted churn counts",pred_churn_counts)



    # print(y_pred)

con_mat=confusion_matrix(y_test,y_pred)
acc_log=accuracy_score(y_test,y_pred)
class_report_logreg=classification_report(y_test,y_pred)


#DECISION TREE CLASSIFIER

decision_classif=DecisionTreeClassifier()
decision_classif.fit(X_train,y_train)

y_preddt=decision_classif.predict(X_test)

confumetr_dectree=confusion_matrix(y_test,y_preddt)
acc_dectree=accuracy_score(y_test,y_preddt)
class_report_Dectree=classification_report(y_test,y_preddt)


    #Random forest classifier

random_forest_model=RandomForestClassifier()
random_forest_model.fit(X_train,y_train)

y_predRF=random_forest_model.predict(X_test)

confumetr_randfor=confusion_matrix(y_test,y_predRF)
acc_ranfor=accuracy_score(y_test,y_predRF)
class_report_ranfor=classification_report(y_test,y_predRF)

    #XGBoost classifier

xgboost_model=XGBClassifier(n_estimators=100,learning_rate=0.1,use_label_encoder=False,eval_metric='logloss')
xgboost_model.fit(X_train,y_train)

y_predxg=xgboost_model.predict(X_test)

confu_metr_xg=confusion_matrix(y_test,y_predxg)
acc_xg=accuracy_score(y_test,y_predxg)
class_report_xgboost=classification_report(y_test,y_predxg)

print("-------------------------Predicted Churn details for Each Model-------------------------")

pred_churn_counts = pd.Series(y_pred).value_counts().sort_index()
print("chrun logistic regression-predicted\n",pred_churn_counts)
print("-------------------------")
pred_churn_counts_dt = pd.Series(y_preddt).value_counts().sort_index()
print("chrun Decision Tree classifier-predicted\n",pred_churn_counts_dt)
print("-------------------------")
pred_churn_counts_rf = pd.Series(y_predRF).value_counts().sort_index()
print("chrun randomforest classifier-predicted\n",pred_churn_counts_rf)
print("-------------------------")
pred_churn_counts_xg = pd.Series(y_predxg).value_counts().sort_index()
print("chrun xgboost classifier-predicted\n",pred_churn_counts_xg)
print("-------------------------")

    #VISUALISATION
fig2,axes=plt.subplots(2,2,figsize=(10,8))

    #confusion matrix plot
sns.heatmap(con_mat,annot=True,fmt='d',cmap='Blues',ax=axes[0,0])
axes[0,0].set_title(f"Logistic regression model-confusion materics\n\n")
axes[0,0].set_xlabel("predicted")
axes[0,0].set_ylabel("actual")


sns.heatmap(confumetr_dectree,annot=True,fmt='d',cmap='Blues',ax=axes[0,1])
axes[0,1].set_title(f"Decision Tree regression model-confusion materics\n\n")
axes[0,1].set_xlabel("predicted")
axes[0,1].set_ylabel("actual")

sns.heatmap(confumetr_randfor,annot=True,fmt='d',cmap='Blues',ax=axes[1,0])
axes[1,0].set_title(f"Randomforest model-confusion matrics\n\n")
axes[1,0].set_xlabel("predicted")
axes[1,0].set_ylabel("actual")

sns.heatmap(confu_metr_xg,annot=True,fmt='d',cmap='Blues',ax=axes[1,1])
axes[1,1].set_title(f"Xgboost classifier\n\n")
axes[1,1].set_xlabel("predicted")
axes[1,1].set_ylabel("actual")
plt.tight_layout()
plt.show(block=False)

    #classification report
fig = plt.figure(figsize=(18, 10))
axes1 = fig.add_subplot(3,2,1)

pos = axes1.get_position()
axes1.set_position([pos.x0 - 0.05, pos.y0, pos.width, pos.height])

fig.suptitle("Customer Churn Dashboard-Summary",fontweight='bold',fontsize=22,
        color="white",
        backgroundcolor="#003366")

    #logistic reg classification report

combined_report = (
        "Logistic Regression:\n\n" + class_report_logreg +
        "\nDecision Tree:\n\n" + class_report_Dectree +
        "\nRandom Forest:\n\n" + class_report_ranfor +
        "\nXGBoost:\n\n" + class_report_xgboost
    )
axes1.text(0.01,0.99,combined_report,fontsize=9,family='monospace',va='top')
axes1.set_title("Classification Reports\n",fontweight='bold',ha='center')
axes1.axis('off')


    #accuracy comparision
models=['Logistric regression','Decsion tree classifier','Random forest classifier','Xgboost classifier']
accuracies_of_model=[acc_log,acc_dectree,acc_ranfor,acc_xg]
axes2 = plt.subplot2grid((2,3), (0,1), colspan=2)


axes2.bar(models,accuracies_of_model)
axes2.set_ylabel("accuracy")
axes2.set_title("Model comparision",fontweight='bold')
axes2.set_ylim(0,1)
    # axes2.set_xticks(range(len(models)))


for i,v in enumerate(accuracies_of_model):
    plt.text(i,v + 0.01,f"{v:2f}",ha='center')


    # Get the probability scores for the positive class (churn = 1)
y_predxg_proba = xgboost_model.predict_proba(X_test)[:, 1]
y_pred_rf_proba = random_forest_model.predict_proba(X_test)[:, 1]
y_pred_dt_proba = decision_classif.predict_proba(X_test)[:, 1]
y_pred_log_proba = logistic_reg_model.predict_proba(X_test)[:, 1]

    # Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_log_proba)

    # Compute AUC score (optional, but useful)

auc_log = roc_auc_score(y_test, y_pred_log_proba)
auc_dt = roc_auc_score(y_test, y_pred_dt_proba)
auc_rf = roc_auc_score(y_test, y_pred_rf_proba)
auc_xg= roc_auc_score(y_test, y_predxg_proba)
print("-------------------------Accuracy of Each Model-------------------------")
print("Logistic Regression AUC:\n", auc_log)
print("-------------------------")
print("Decision Tree AUC:\n", auc_dt)
print("-------------------------")
print("Random Forest AUC:\n", auc_rf)
print("-------------------------")
print("XGBoost AUC:\n", auc_xg)
print("-------------------------")
    # Plot ROC curve
axes3 = fig.add_subplot(2,3,5)
axes3.plot(fpr, tpr, label='Logistic ROC Curve (AUC = %.2f)' % auc_log)
axes3.plot([0, 1], [0, 1], label='Random')
axes3.set_xlabel("False Positive Rate")
axes3.set_ylabel("True Positive Rate")
axes3.set_title("Logistic ROC Curve",fontweight='bold')

axes3.legend(fontsize=8,loc='lower right')

    #count_plot for actual churn
axes4 = fig.add_subplot(2,3,6)
pos = axes4.get_position()
axes4.set_position([pos.x0 + 0.03, pos.y0 + 0.04, pos.width - 0.12, pos.height - 0.10])

sns.barplot(x=['No','Yes'], y=Churn_actual, ax=axes4,palette=["#0DF018", "#B71301"],hue=[0,1])
axes4.set_title("Actual Churn Count\n",fontweight='bold')
axes4.set_xlabel("Churn")
axes4.set_ylabel("Count")

axes4.legend([f"Raw Data Churn= {y.shape[0]}"],
                bbox_to_anchor=(-0.10, -0.35), loc='upper left')
    # axes4.axis("off")
for i, v in enumerate(Churn_actual):
        axes4.text(i, v + 300, str(v), ha='center', fontsize=9, fontweight='bold')


    # Logistic Regression Predicted Churn Count Plot
axes5 = fig.add_subplot(2,3,6)
Testdata=y_test.shape[0]
    # Move the predicted plot a little up & left (optional tuning)
pos = axes5.get_position()
axes5.set_position([pos.x0 + 0.20, pos.y0 + 0.04, pos.width - 0.12, pos.height - 0.10])

sns.barplot(x=[0,1], y=pred_churn_counts, ax=axes5, palette=["#0DF018", "#B71301"],hue=[0, 1])
axes5.set_title("Predicted Churn Count\n(Logistic Regression)\n",fontweight='bold')
axes5.set_xlabel("Churn Prediction\n(0=No, 1=Yes)")
axes5.set_ylabel("Count")
axes5.legend([f"Test Data= {Testdata}"],
                bbox_to_anchor=(0.0, -0.35), loc='upper left')

for i, v in enumerate(pred_churn_counts):
        axes5.text(i, v + 80, str(v), ha='center', fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()

