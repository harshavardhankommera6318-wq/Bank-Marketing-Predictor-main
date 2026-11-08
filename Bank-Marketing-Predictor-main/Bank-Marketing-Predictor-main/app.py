from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.data_cleaning import clean_data,TARGET
from src.model import train_model,predict_subscription
st.set_page_config(page_title="Bank Marketing Predictor",page_icon="🏦",layout="wide")
sns.set_theme(style="whitegrid")
st.title("🏦 Bank Marketing Predictor")
st.caption("Predictive classification demonstration using a synthetic campaign dataset. Not for real banking decisions.")
@st.cache_data
def load_data(path): return pd.read_csv(path)
@st.cache_resource
def fit(data): return train_model(data)
try:
    data=clean_data(load_data(str(ROOT/"data"/"bank_marketing.csv")))
    model,X_test,y_test,metrics=fit(data)
except Exception as e:
    st.error(f"Unable to load or train: {e}"); st.stop()
with st.sidebar:
    st.header("Filters")
    jobs=st.multiselect("Job category",sorted(data.job.unique()),default=sorted(data.job.unique()))
    st.caption("1,200 synthetic customer records")
view=data[data.job.isin(jobs)]
yes=int((data[TARGET]=="yes").sum())
a,b,c,d=st.columns(4)
a.metric("Customers",f"{len(data):,}"); b.metric("Subscribed",f"{yes:,}")
c.metric("Subscription rate",f"{yes/len(data)*100:.1f}%"); d.metric("Model F1",f"{metrics['f1']:.3f}")
overview,records,predict,evaluation=st.tabs(["Campaign overview","Customer records","Predict subscription","Model evaluation"])
with overview:
    l,r=st.columns(2)
    with l:
        fig,ax=plt.subplots(); sns.countplot(data=view,x=TARGET,hue=TARGET,legend=False,ax=ax); ax.set_title("Subscription outcomes"); st.pyplot(fig); plt.close(fig)
    with r:
        rates=view.groupby("job")[TARGET].apply(lambda s:(s=="yes").mean()*100).sort_values()
        fig,ax=plt.subplots(); rates.plot(kind="barh",ax=ax); ax.set_title("Subscription rate by job (%)"); st.pyplot(fig); plt.close(fig)
    fig,ax=plt.subplots(); sns.histplot(data=view,x="age",hue=TARGET,multiple="stack",ax=ax); ax.set_title("Age distribution by outcome"); st.pyplot(fig); plt.close(fig)
with records:
    st.dataframe(view,use_container_width=True,hide_index=True)
    st.download_button("Download filtered CSV",view.to_csv(index=False).encode(), "bank_marketing_filtered.csv","text/csv")
with predict:
    st.subheader("Estimate subscription likelihood")
    with st.form("customer"):
        x1,x2,x3=st.columns(3)
        with x1:
            age=st.slider("Age",18,90,35); job=st.selectbox("Job",sorted(data.job.unique()))
            marital=st.selectbox("Marital status",["married","single","divorced"])
            education=st.selectbox("Education",["primary","secondary","tertiary","unknown"])
            balance=st.number_input("Account balance",-5000,50000,1200,100)
        with x2:
            housing=st.selectbox("Housing loan",["yes","no"]); personal=st.selectbox("Personal loan",["yes","no"])
            contact=st.selectbox("Contact type",["cellular","telephone","unknown"])
            month=st.selectbox("Last contact month",["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
        with x3:
            campaign=st.slider("Contacts in this campaign",1,30,2)
            pdays=st.selectbox("Days since previous contact",[-1,5,10,15,30,60,90,120,180])
            previous=st.slider("Previous campaign contacts",0,20,0)
        submitted=st.form_submit_button("Predict subscription",type="primary")
    if submitted:
        customer={"age":age,"job":job,"marital":marital,"education":education,"account_balance":balance,"housing_loan":housing,"personal_loan":personal,"contact_type":contact,"last_contact_month":month,"campaign_contacts":campaign,"days_since_previous_contact":pdays,"previous_contacts":previous}
        label,p=predict_subscription(model,customer)
        st.metric("Estimated probability",f"{p*100:.1f}%")
        st.success("Model prediction: YES") if label=="yes" else st.info("Model prediction: NO")
        st.caption("This probability is a synthetic-data model estimate, not a guarantee or customer recommendation.")
with evaluation:
    cols=st.columns(5)
    for col,(name,value) in zip(cols,metrics.items()): col.metric(name.replace("_"," ").title(),f"{value:.3f}")
    st.write("80/20 stratified train/test split. Logistic Regression; numeric standardization; categorical one-hot encoding.")
    st.warning("Synthetic educational data only. Do not use for credit, lending, eligibility, or real customer treatment decisions.")
