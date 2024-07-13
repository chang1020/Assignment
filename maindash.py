import streamlit as st
import sqlite3
import prediction as sp

def create_connection():
    conn = sqlite3.connect('stocks.db')
    return conn

def create_table(conn):
    with conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS stocks
                        (date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER, ticker TEXT)''')

def maindash():
    st.sidebar.title("Menu")
    selection = st.sidebar.selectbox(
        "Go to",
        ("Predict Stock Price",)
    )
    conn = create_connection()
    create_table(conn)
    ticker = None
    if selection == "Predict Stock Price":
        sp.page2(conn, ticker)
    conn.close()

if __name__ == "__main__":
    maindash()
