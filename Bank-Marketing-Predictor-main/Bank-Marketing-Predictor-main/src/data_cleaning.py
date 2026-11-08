import pandas as pd
TARGET="subscribed"
NUMERIC_FEATURES=["age","account_balance","campaign_contacts","days_since_previous_contact","previous_contacts"]
CATEGORICAL_FEATURES=["job","marital","education","housing_loan","personal_loan","contact_type","last_contact_month"]
FEATURES=NUMERIC_FEATURES+CATEGORICAL_FEATURES
REQUIRED=FEATURES+[TARGET]
def clean_data(frame):
    missing=[c for c in REQUIRED if c not in frame.columns]
    if missing: raise ValueError(f"Missing required columns: {', '.join(missing)}")
    data=frame[REQUIRED].copy()
    for c in NUMERIC_FEATURES:
        data[c]=pd.to_numeric(data[c],errors="coerce")
        data[c]=data[c].fillna(data[c].median())
    for c in CATEGORICAL_FEATURES:
        data[c]=data[c].astype("string").fillna("unknown").str.strip().str.lower().replace("","unknown")
    data[TARGET]=data[TARGET].astype("string").str.strip().str.lower()
    data=data[data[TARGET].isin(["yes","no"])].dropna(subset=NUMERIC_FEATURES)
    data["age"]=data["age"].clip(18,100)
    data["campaign_contacts"]=data["campaign_contacts"].clip(1,100)
    data["previous_contacts"]=data["previous_contacts"].clip(0,100)
    data=data.drop_duplicates().reset_index(drop=True)
    if len(data)<30: raise ValueError("At least 30 valid rows are required.")
    return data
