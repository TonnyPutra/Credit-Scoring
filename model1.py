def run():
    # %% Credit Scoring Dashboard.ipynb 2
    import joblib
    import requests
    import numpy as np
    import pandas as pd
    import streamlit as st
    import matplotlib.pyplot as plt
    import streamlit_authenticator as stauth 
    from datetime import datetime
    from tensorflow.keras.models import load_model

    API_KEY = "cur_live_lN0DkPaOPRfRetRKByzfT38fG0vuQa5CIJjBuORv"
    BASE_URL = "https://api.currencyapi.com/v3/latest"

    @st.cache_data(ttl=86400)  # Cache data for 24 hours (86400 seconds)
    def get_exchange_rate():
        """Fetches and caches the IDR to USD exchange rate for 24 hours."""
        params = {
            "apikey": API_KEY,
            "base_currency": "IDR",
            "currencies": "USD"
        }
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if "data" in data and "USD" in data["data"]:
            exchange_rate = data["data"]["USD"]["value"]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return exchange_rate, timestamp  # Return rate + timestamp
        else:
            st.error("Failed to fetch exchange rate. Please try again later.")
            return None, None

    # %% Credit Scoring Dashboard.ipynb 5
    # st.set_page_config(layout="wide")

    st.markdown("""
      <style>
          /* Move content higher */
          .block-container {
              padding-top: 15px !important;  /* Reduce top padding */
          }
          
          /* Button styling */
          .stButton>button {
              background-color: #123524;
              color: white;
              border-radius: 5px;
              padding: 10px 20px;
              font-size: 16px;
          }

          /* Add background color to the container holding the columns */
          .st-emotion-cache-1wmy9hl > div {
              background-color: #185519 !important;  /* Set the column background color */
              margin: 7px 7px 7px 7px;
          }

          .st-emotion-cache-434r0z > div:first-child {
              margin-right: 10px;
          }
          
          .st-emotion-cache-434r0z > div:last-child {
              margin-right: 17px;
          }

          footer {visibility: hidden;}

          .st-emotion-cache-o4xmfe {
              width: 250px; /* Set custom width */
              height: auto; /* Maintain aspect ratio */
          }
      </style>
    """, unsafe_allow_html=True)

    st.logo("logo.png", link="https://ptatkb.idjams.com/", size="large")
    # %% Credit Scoring Dashboard.ipynb 7
    scaler = joblib.load('train_scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    model = load_model('model.h5')

    # Mapping Categorical Features
    home_ownership_map = {
      "SEWA": "RENT",
      "KPR": "MORTGAGE",
      "MILIK SENDIRI": "OWN",
      "LAINNYA": "OTHER"
    }

    loan_intent_map = {
      "PENDIDIKAN": "EDUCATION",
      "MEDIS": "MEDICAL",
      "USAHA": "VENTURE",
      "PRIBADI": "PERSONAL",
      "KONSOLIDASI HUTANG": "DEBTCONSOLIDATION",
      "PERBAIKAN RUMAH": "HOMEIMPROVEMENT"
    }

    default_history_map = {"Y": "Y", "N": "N"}  # Tetap sama
    loan_grade_map = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G"}  # Tetap sama

    # %% Credit Scoring Dashboard.ipynb 8
    st.markdown("**Masukkan Informasi Pemohon**")
    i1, i2, i3 = st.columns(3)
    rate, last_updated = get_exchange_rate()
    with i1: 
        person_age = st.number_input("Usia Pemohon", min_value=18, max_value=75, value=25, placeholder="Masukkan nilai")
        person_income = st.number_input("Pendapatan Pemohon (IDR)", min_value=10000000, max_value=1000000000, value=10000000, placeholder="Masukkan nilai")
        person_emp_length = st.number_input("Lama Bekerja (tahun)", min_value=0, max_value=50, value=15, placeholder="Masukkan nilai")
        person_home_ownership_id = st.selectbox(
            "Status Kepemilikan Rumah", list(home_ownership_map.keys()), index=2, placeholder="Masukkan pilihan"
        )

    with i2:
        loan_amnt = st.number_input("Jumlah Pinjaman (IDR)", min_value=1000000, max_value=2000000000, value=1000000, placeholder="Masukkan nilai")
        loan_intent_id = st.selectbox(
            "Tujuan Pinjaman", list(loan_intent_map.keys()), index=1, placeholder="Masukkan pilihan"
        )
        loan_int_rate = st.number_input("Suku Bunga Pinjaman (%)", min_value=0.0, max_value=25.0, value=13.31, placeholder="Masukkan nilai")
        loan_percent_income = st.number_input("Persentase Pendapatan untuk Pinjaman", min_value=0.0, max_value=1.0, value=0.67, placeholder="Masukkan nilai")
    with i3: 
        cb_person_cred_hist_length = st.number_input("Lama Riwayat Kredit (tahun)", min_value=0, max_value=30, value=0, placeholder="Masukkan nilai")
        loan_grade_id = st.selectbox("Grade Pinjaman", list(loan_grade_map.keys()), index=2, placeholder="Masukkan pilihan")
        cb_person_default_on_file_id = st.selectbox("Riwayat Kredit Buruk (Y/N)", list(default_history_map.keys()), index=1, placeholder="Masukkan pilihan")    

    person_home_ownership = home_ownership_map[person_home_ownership_id]
    loan_intent = loan_intent_map[loan_intent_id]
    loan_grade = loan_grade_map[loan_grade_id]
    cb_person_default_on_file = default_history_map[cb_person_default_on_file_id]

    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income*rate],
        'person_emp_length': [person_emp_length],
        'loan_amnt': [loan_amnt*rate],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        'person_home_ownership': [person_home_ownership],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'cb_person_default_on_file': [cb_person_default_on_file]
    })


    numerical_features = ['person_age', 'person_income', 'person_emp_length',
                          'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                          'cb_person_cred_hist_length']
    categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade',
                            'cb_person_default_on_file']


    input_data[numerical_features] = scaler.transform(input_data[numerical_features])


    for col in categorical_features:
        input_data[col] = label_encoders[col].transform(input_data[col])


    input_dict = {f"{feature}_input": input_data[feature].values.reshape(-1, 1) for feature in categorical_features}
    input_dict["numerical_input"] = input_data[numerical_features].values

    c1, c2 = st.columns([1, 4])  # Adjust width as needed

    with c1:
        if st.button("Prediksi Status Pinjaman"):
            prediction = model.predict(input_dict)
            predicted_class = "DITOLAK" if prediction[0][0] >= 0.5 else "DITERIMA"
            with c2:  # Place result in second column
                st.subheader(f"Hasil Prediksi Pinjaman: **{predicted_class}**")