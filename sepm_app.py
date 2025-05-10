import streamlit as st
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import base64
import time
from datetime import datetime, timedelta
import io
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# SETUP
st.set_page_config(page_title="Smart Engineering Project Manager", layout="wide")
if 'tasks' not in st.session_state:
    st.session_state.tasks = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

st.title("🚀 Smart Engineering Project Manager (SEPM)")

menu = st.sidebar.selectbox("Navigate",
                            ["🏠 Home", "📁 Project Manager", "📐 Engineering Toolkit", "📊 Gantt Chart", "📄 Report Viewer",
                             "🤖 AI Assistant"])

# HOME
if menu == "🏠 Home":
    st.markdown("### Welcome to SEPM — Your All-in-One Engineering Workflow App!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📂 Active Projects", len(st.session_state.tasks))
    with col2:
        st.metric("📎 Files Uploaded", len(st.session_state.uploaded_files))
    with col3:
        st.metric("⚙️ Tools Used", 3)
    st.info("Use the sidebar to upload tasks, calculate equations, visualize reports, and get help from AI.")

# PROJECT MANAGER
elif menu == "📁 Project Manager":
    st.subheader("📁 Project Manager")
    with st.form("task_form"):
        name = st.text_input("Task Name")
        deadline = st.date_input("Deadline")
        status = st.selectbox("Status", ["Not Started", "In Progress", "Completed"])
        submitted = st.form_submit_button("Add Task")
        if submitted and name:
            st.session_state.tasks.append({"name": name, "deadline": deadline, "status": status})
            st.success("✅ Task added!")

    if st.session_state.tasks:
        st.markdown("### 📝 Task List")
        df = pd.DataFrame(st.session_state.tasks)
        st.dataframe(df)

    st.markdown("### 📤 Upload Files")
    files = st.file_uploader("Upload project files", accept_multiple_files=True)
    if files:
        st.session_state.uploaded_files += files
        st.success("Files uploaded!")

# TOOLKIT
elif menu == "📐 Engineering Toolkit":
    st.subheader("🧮 Engineering Calculator (Symbolic + Numeric)")
    calc_type = st.radio("Choose mode:", ["Symbolic", "Numeric"])

    if calc_type == "Symbolic":
        expr = st.text_input("Enter equation (e.g., x**2 + 2*x + 1)")
        var = st.text_input("Differentiate w.r.t. (e.g., x)")
        if st.button("Differentiate"):
            try:
                x = sp.Symbol(var)
                res = sp.diff(expr, x)
                st.success(f"Result: {res}")
            except:
                st.error("Invalid expression.")
    else:
        f = st.text_input("Enter function (e.g., sin(x) * exp(x))")
        domain = st.slider("X range", -10, 10, (-5, 5))
        if st.button("Plot Function"):
            try:
                x = np.linspace(domain[0], domain[1], 400)
                y = eval(f"np.{f}")
                plt.plot(x, y)
                st.pyplot(plt.gcf())
            except Exception as e:
                st.error(f"Error: {e}")

# GANTT CHART
elif menu == "📊 Gantt Chart":
    st.subheader("📅 Auto-Generated Gantt Chart")
    if st.session_state.tasks:
        df = pd.DataFrame(st.session_state.tasks)
        df['Start'] = pd.to_datetime(datetime.today())
        df['End'] = pd.to_datetime(df['deadline'])
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, row in df.iterrows():
            ax.barh(row['name'], (row['End'] - row['Start']).days, left=row['Start'], color='skyblue')
        ax.set_xlabel("Date")
        ax.set_ylabel("Task")
        st.pyplot(fig)
    else:
        st.warning("No tasks added yet.")

# REPORT VIEWER
elif menu == "📄 Report Viewer":
    st.subheader("📄 Report/Simulation Viewer")
    uploaded = st.file_uploader("Upload a CSV, PDF, or TXT file", type=["csv", "pdf", "txt"])
    if uploaded:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
            st.dataframe(df)
        elif uploaded.name.endswith(".txt"):
            content = uploaded.read().decode()
            st.text_area("Contents", content, height=300)
        elif uploaded.name.endswith(".pdf"):
            base64_pdf = base64.b64encode(uploaded.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

# AI ASSISTANT
elif menu == "🤖 AI Assistant":
    st.subheader("💬 Ask the Engineering Assistant (GPT-powered)")
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    question = st.text_area("Ask a technical question:")
    if st.button("Get Answer"):
        if not api_key or not question:
            st.warning("Please enter your API key and question.")
        else:
            try:

                openai.api_key = api_key
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a smart engineering assistant."},
                        {"role": "user", "content": question},
                    ]
                )
                answer = response['choices'][0]['message']['content']
                st.success("Answer:")
                st.markdown(answer)
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.markdown("""
  <center>
      🔧 Developed with passion by **Praise Adeyeye**  
      🧠 _"Engineering isn’t just equations — it’s imagination made real."_ 💡  
      <br>
      © 2025 Praise Adeyeye. All rights reserved. 🚀
  </center>
  """, unsafe_allow_html=True)

