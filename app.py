import streamlit as st
import pandas as pd
import joblib
import re
import io
import datetime
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('titanic_model.pkl')
MODEL_VERSION = "1.0 (trained on 2025-07-08)"

st.title("🚢 Titanic Survival Prediction App")

st.markdown("""
 **TIP:** The survival prediction shows a percentage (confidence level).  
A score above **50%** means the passenger is likely to have **survived**.  
The more accurately you fill the details (age, fare, name title, etc.),  
the better the prediction result.  
""")

st.markdown("**Fields marked with an asterisk (*) are required and must be filled correctly.**")
st.markdown("**NOTE:** If Age is entered as 0, it will default to 28 (median age). Embarked defaults to 'S' (most common port).")

# --- Initialize history DataFrame in session_state ---
if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=[
        'Name', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
        'Has_Cabin', 'FamilySize', 'Sex', 'Embarked',
        'Title', 'Prediction', 'Confidence', 'Timestamp'
    ])

# --- User Inputs ---
Name = st.text_input("Full Name (e.g. Braund, Mr. Owen Harris) *", value="", help="Enter full passenger name including title")
Pclass = st.selectbox("Passenger Class *", [1, 2, 3], index=2, help="1 = 1st class, 2 = 2nd class, 3 = 3rd class")
Sex = st.selectbox("Sex *", ['male', 'female'], index=0)
Age = st.number_input("Age *", min_value=0, max_value=100, value=28, step=1, help="Enter age as a whole number. Zero will be replaced with median age (28).")
SibSp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0, step=1)
Parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0, step=1)
Fare = st.number_input("Fare *", min_value=0.0, value=10.5, step=0.1, format="%.2f", help="Fare paid by passenger, in British Pounds")
Embarked = st.selectbox("Port of Embarkation *", ['S', 'C', 'Q'], index=0, help="S = Southampton, C = Cherbourg, Q = Queenstown")
Has_Cabin = st.radio("Did the passenger have a cabin? *", ['Yes', 'No'], index=1, help="Indicate if passenger had a recorded cabin number")

# --- Feature Engineering ---
def extract_title(name):
    match = re.search(r',\s*([^\.]*)\.', name)
    return match.group(1).strip() if match else "Other"

def map_title(title):
    return {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs"}.get(title, "Other")

def preprocess_input():
    formatted_name = Name.strip().title()
    title_raw = extract_title(formatted_name)
    Title = map_title(title_raw)

    final_age = 28 if Age == 0 else Age
    FamilySize = SibSp + Parch + 1
    has_cabin_value = 1 if Has_Cabin == 'Yes' else 0
    sex_male = 1 if Sex == 'male' else 0
    embarked_Q = 1 if Embarked == 'Q' else 0
    embarked_S = 1 if Embarked == 'S' else 0

    data = {
        'Pclass': Pclass,
        'Age': final_age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Has_Cabin': has_cabin_value,
        'FamilySize': FamilySize,
        'Sex_male': sex_male,
        'Embarked_Q': embarked_Q,
        'Embarked_S': embarked_S,
        'Title_Miss': 0,
        'Title_Mr': 0,
        'Title_Mrs': 0,
        'Title_Other': 0
    }

    title_column = f"Title_{Title}"
    data[title_column] = 1 if title_column in data else 1
    return pd.DataFrame([data])

# --- Validation ---
def validate_inputs():
    errors = []
    if not Name.strip():
        errors.append("Name cannot be empty.")
    if Fare <= 0:
        errors.append("Fare must be greater than zero.")
    if Age > 100 or Age < 0:
        errors.append("Age must be between 0 and 100.")
    return errors

# --- Main Prediction block ---
if st.button("Predict"):
    errors = validate_inputs()
    if errors:
        for err in errors:
            st.error(f"⚠️ {err}")
    else:
        input_df = preprocess_input()
        ordered_columns = [
            'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
            'Has_Cabin', 'FamilySize', 'Sex_male',
            'Embarked_Q', 'Embarked_S',
            'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other'
        ]
        input_df = input_df[ordered_columns]

        prediction = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df)[0][1]
        result = "Would Have Survived" if prediction == 1 else "Would Not Have Survived"

        st.success(f"✅ The passenger: **{result}**")
        st.info(f"🔍 Model confidence: **{confidence:.2%}**")
        st.progress(confidence)

        fig, ax = plt.subplots()
        ax.pie([confidence, 1-confidence],
               labels=["Survived", "Did Not Survive"],
               autopct='%1.1f%%',
               colors=['#81C784', '#E57373'])
        ax.set_title("Survival Probability Distribution", fontsize=12)
        st.pyplot(fig)
        st.caption("🟩 Softer Green = Survived | 🟥 Softer Red = Did Not Survive")

        # --- Append to history ---
        new_entry = {
            'Name': Name.strip().title(),
            'Pclass': Pclass,
            'Age': 28 if Age == 0 else Age,
            'SibSp': SibSp,
            'Parch': Parch,
            'Fare': Fare,
            'Has_Cabin': 1 if Has_Cabin == 'Yes' else 0,
            'FamilySize': SibSp + Parch + 1,
            'Sex': Sex,
            'Embarked': Embarked,
            'Title': map_title(extract_title(Name)),
            'Prediction': result,
            'Confidence': confidence,
            'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state['history'] = pd.concat([
            st.session_state['history'],
            pd.DataFrame([new_entry])
        ], ignore_index=True)

        # --- Show history ---
        st.markdown("### 📜 Prediction History")
        history_df = st.session_state['history']
        st.dataframe(history_df)

        if not history_df.empty:
            history_df['label'] = history_df['Name'] + " | " + history_df['Timestamp']
            selected_label = st.selectbox(
                "Select one entry from history to download:",
                options=history_df['label']
            )

            selected_row = history_df[history_df['label'] == selected_label].drop(columns=['label'])
            single_csv = io.StringIO()
            selected_row.to_csv(single_csv, index=False)
            st.download_button(
                label="Download selected entry as CSV",
                data=single_csv.getvalue(),
                file_name=f"titanic_prediction_{selected_label.replace(' | ', '_').replace(':','-')}.csv",
                mime="text/csv"
            )

            full_csv = io.StringIO()
            history_df.drop(columns=['label']).to_csv(full_csv, index=False)
            st.download_button(
                label="Download full history as CSV",
                data=full_csv.getvalue(),
                file_name="titanic_prediction_history.csv",
                mime="text/csv"
            )

# --- Footer ---
st.markdown(f"---\nModel Version: **{MODEL_VERSION}**")
