import pickle
from pathlib import Path

import streamlit_authenticator as stauth

names = ["Mr. Stark"]
usernames = ["Tstark"]
passwords = ["314159"]
email = ["mr.stark@example.com"]

# Create a Hasher object
hasher = stauth.Hasher(passwords)

# Hash the passwords individually and store in a list
hashed_passwords = [hasher.hash(password) for password in passwords] 

# Specify the desired file path directly
file_path = Path("hashed_pw.pkl")  # or Path("./hashed_pw.pkl") for current directory
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)