import streamlit as st
import json
import requests

st.title("Test API")
if st.button("testAPI"):
    res = requests.get(url="http://127.0.0.1:8000/")
    st.subheader({res.text})

