# AUTOGENERATED! DO NOT EDIT! File to edit: Credit Scoring Dashboard.ipynb.

# %% auto 0
__all__ = []

# %% Credit Scoring Dashboard.ipynb 2
import joblib
import streamlit as st
import streamlit_authenticator as stauth 
import model1

# %% Credit Scoring Dashboard.ipynb 5
st.set_page_config(layout="wide")

st.markdown("""
    <style>
        .st-emotion-cache-16h9saz {
            background: #185519 !important;
        }
        
        .st-emotion-cache-1pst7dz {
            background-color: #123524 !important;
        }

        .st-au {
            background-color: #123524 !important;
        }

        [data-testid="stHeader"] div:nth-child(2) {
            display: none;
        }
    </style>

""", unsafe_allow_html=True)


hashed_passwords = joblib.load('hashed_pw.pkl')

credentials = {
    "usernames": {
        "Tstark": {  # Username as key
            "email": "mr.stark@example.com",
            "name": "Mr. Stark",
            "password": hashed_passwords[0]
        }
    }
}

# Initialize authenticator
authenticator = stauth.Authenticate(
    credentials,  
    "credit_scoring_dashboard",  
    "yke",  
    cookie_expiry_days=7  
)

login_result = authenticator.login(location="main")

# Check session state directly
if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
  # Apply custom CSS
  st.markdown("""
      <style>
          /* Move content higher */
          .block-container {
              padding: 15px;
              max-width: 1000px !important;  /* Adjust as needed */
              margin: auto !important;       /* Centers content */
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
              padding: 15px;
              margin-left: -15px;
          }


          # header {visibility: hidden;}
          footer {visibility: hidden;}

          .st-emotion-cache-o4xmfe {
              width: 250px; /* Set custom width */
              height: auto; /* Maintain aspect ratio */
          }

        .st-emotion-cache-cit1en {
            position: absolute;
            bottom: -280px; /* Default value */
            left: 50%;
            transform: translateX(-50%);
            text-align: center;
            transition: bottom 0.3s ease-in-out; /* Smooth transition */
        }

        /* Hide all header icons except the logo */
        [data-testid="stHeader"] div:nth-child(2) {
            display: none;
        }

        .st-emotion-cache-5drf04 {
            max-width: 15rem; /* Set custom width */
            height: auto; /* Maintain aspect ratio */
        }

        .st-emotion-cache-1hvtlnw {
            height: 5.75rem;
        }

        .st-emotion-cache-6qob1r {
            background-color: #185519 !important; 
        }
      </style>

      <script>
          function adjustLogoutPosition() {
              let sidebar = document.querySelector('[data-testid="stSidebar"]');
              let logoutButton = document.querySelector('.st-emotion-cache-1q8sxg4');

              if (sidebar && logoutButton) {
                  let sidebarHeight = sidebar.clientHeight; // Get sidebar height
                  let offset = 280;  // Adjust this value as needed
                  logoutButton.style.bottom = -(sidebarHeight * 0.15 + offset) + 'px';
              }
          }
      </script>
  """, unsafe_allow_html=True)
    
  st.logo("logo.png", link="https://ptatkb.idjams.com/")
  with st.sidebar:
      st.sidebar.write("Halo", st.session_state['name'])
      if st.button("🏠 Home", use_container_width=True):
          st.session_state["page"] = "Home"

      if st.button("📊 Model 1", use_container_width=True):
          st.session_state["page"] = "Model 1"
      authenticator.logout(location="sidebar")

  if "page" not in st.session_state:
      st.session_state["page"] = "Home"

  # 🚀 Load the selected page WITHOUT reloading the whole app
  if st.session_state["page"] == "Model 1":
      model1.run()
      st.stop()
  
  # %% Credit Scoring Dashboard.ipynb 6
  st.title("Credit Scoring Dashboard")
  # %% Credit Scoring Dashboard.ipynb 8
  col1, col2 = st.columns(2, gap="medium")

  with col1:
      st.subheader("Apa itu skor kredit?")
      st.markdown("""
          <style>
              .justified-text {
                  text-align: justify;
              }
          </style>
          <div class="justified-text">
              Dalam suatu proses pengajuan pinjaman, kreditur akan menentukan keputusan
              dari pengajuan pinjaman berdasarkan data diri dan riwayat catatan biro kredit. 
              Oleh karena itu, mengetahui kemungkinan keputusan tersebut dapat membantu dalam
              merencanakan peminjaman.
          </div>
      """, unsafe_allow_html=True)
      
  with col2:
      st.subheader("Apa Yang Kami Bawakan?")
      st.markdown("""
          <style>
              .justified-text {
                  text-align: justify;
              }
          </style>
          <div class="justified-text">
              Dengan menggunakan Deep Learning, kami membuat sebuah AI yang dapat memprediksi terkait 
              kredit skor. Model yang saat ini sudah berhasil dibangun menggunakan Neural Network
              bertipe Multi-Layer Perceptron dapat memprediksi keputusan peminjaman yang diajukan
              seseorang berdasarkan beberapa input terkait. Model ini mencapai akurasi hingga 88% 
              pada data latih.
          </div>
      """, unsafe_allow_html=True)

else:
    # If not authenticated, show login form or error
    st.error("You are not logged in.")
