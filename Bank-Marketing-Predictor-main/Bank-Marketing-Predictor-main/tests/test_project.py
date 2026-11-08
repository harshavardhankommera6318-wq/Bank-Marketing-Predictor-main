from pathlib import Path
import pandas as pd
from src.data_cleaning import clean_data
from src.model import train_model,predict_subscription
ROOT=Path(__file__).resolve().parents[1]
def test_cleaning():
    data=clean_data(pd.read_csv(ROOT/"data"/"bank_marketing.csv"))
    assert len(data)>=30 and set(data.subscribed.unique()) <= {"yes","no"}
def test_prediction_probability():
    data=clean_data(pd.read_csv(ROOT/"data"/"bank_marketing.csv"))
    model,_,_,_=train_model(data)
    customer={"age":35,"job":"admin","marital":"single","education":"tertiary","account_balance":1200,"housing_loan":"no","personal_loan":"no","contact_type":"cellular","last_contact_month":"may","campaign_contacts":2,"days_since_previous_contact":-1,"previous_contacts":0}
    label,p=predict_subscription(model,customer)
    assert label in {"yes","no"} and 0<=p<=1
