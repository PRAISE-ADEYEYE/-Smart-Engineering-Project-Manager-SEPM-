import base64
import io
import os
import time
from datetime import datetime
from statistics import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import plotly.express as px
import streamlit as st
import sympy as sp
from jax._src.source_info_util import summarize
from openai import OpenAI
from pygments.lexers import go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import datetime
datetime.datetime.now().isoformat()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
# SETUP
st.set_page_config(page_title="Smart Engineering Project Manager", layout="wide")
if 'tasks' not in st.session_state:
    st.session_state.tasks = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

st.title("🚀 Smart Engineering Project Manager (SEPM)")

menu = st.sidebar.selectbox("Navigate",
                            ["🏠 Home", "📁 Project Manager", "📐 Engineering Toolkit", "📊 Gantt Chart", "📄 Report Viewer",
                             "🤖 AI Assistant", "📈 Data Analyzer", "📋 BOM Manager", "🧠 Calculator",
                             "📘 Engineering Glossary", "📎 Notes & Docs", "🔁 Version Control", "🛠️ Unit Converter",
                             "📦 Inventory Tracker", "🎯 Goal Tracker", "🧪 Test Log", "🕒 Time Tracker",
                             "🖼️ CAD Viewer", "🔍 Error Log Analyzer", "📡 Sensor Monitor", "🧰 Tool Scheduler",
                             "🎵📏 Sound metre pro", "📝✨ Notes summarizer", "🎥🔧 Engineering videos"])


# HOME
def ARIMA():
    pass


if menu == "🏠 Home":
    # ----------  PAGE CONFIG & HERO BANNER  ----------
    st.markdown(
        """
        <style>
        /* Center hero text */
        h1,h3,h4 {text-align:center; color:#fefefefe;}
        /* Divider line tweak */
        hr {border-top: 1px solid rgba(0,255,255,0.3);}
        </style>
        """,
        unsafe_allow_html=True
    )

    # ----------  HERO TYPE‑WRITER  ----------
    hero1 = st.empty()
    msg1 = "### 🚀 **Welcome to SEPM — Your All‑in‑One Engineering Workflow Hub!**"
    typed = ""
    for ch in msg1:
        typed += ch
        hero1.markdown(typed)
        time.sleep(0.03)

    hero2 = st.empty()
    msg2 = "##### 👋 Hey Engineer, here’s your live mission control panel:"
    typed = ""
    for ch in msg2:
        typed += ch
        hero2.markdown(typed)
        time.sleep(0.03)

    # ----------  TOP‑LEVEL KPIs  ----------
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("📂 Active Projects", len(st.session_state.tasks))
    kpi2.metric("📎 Files Uploaded", len(st.session_state.uploaded_files))
    kpi3.metric("🛠️ Modules Available", 24)

    st.markdown("---")

    # ----------  MINI DASHBOARD  ----------
    left, right = st.columns(2)

    with left:
        st.subheader("📊 Task Status Snapshot")
        status_data = pd.DataFrame({
            "Status": ["Completed", "Pending"],
            "Count": [
                sum(t["status"] == "Completed" for t in st.session_state.tasks),
                sum(t["status"] != "Completed" for t in st.session_state.tasks)
            ]
        })
        fig_status = px.pie(
            status_data,
            names="Status",
            values="Count",
            hole=0.45,
            color_discrete_sequence=["#00d1b2", "#ffdd57"],
        )
        fig_status.update_traces(textposition='inside', textinfo='percent+label')
        fig_status.update_layout(margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_status, use_container_width=True)

    with right:
        st.subheader("📈 Weekly Productivity")
        prod_df = pd.DataFrame({
            "Day": [f"Day {i}" for i in range(1, 8)],
            "Hours": np.random.randint(2, 9, 7)
        })
        fig_line = px.area(
            prod_df, x="Day", y="Hours",
            markers=True,
            color_discrete_sequence=["#3273dc"]
        )
        fig_line.update_layout(yaxis_title="Hours Worked",
                               xaxis_title=None,
                               margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    st.success(
        "💡 **Pro Tip:** use the sidebar to explore goal tracking, inventory tools, sensor monitors, AI summarization, and much more!"
    )
    st.markdown(
        "<div style='text-align:center; color:#fefefe; font-style:italic;'>"
        "Stay consistent, stay smart — SEPM keeps you building like a pro 🧠✨"
        "</div>",
        unsafe_allow_html=True
    )
# PROJECT MANAGER
elif menu == "📁 Project Manager":
    st.subheader("📁 Project Manager")

    # Initialize tasks and uploads
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

    # Task Form
    with st.form("task_form"):
        name = st.text_input("📝 Task Name")
        deadline = st.date_input("📅 Deadline")
        status = st.selectbox("📌 Status", ["Not Started", "In Progress", "Completed"])
        submitted = st.form_submit_button("➕ Add Task")

        if submitted and name:
            task_id = len(st.session_state.tasks) + 1
            st.session_state.tasks.append({
                "ID": task_id,
                "Task Name": name,
                "Deadline": deadline,
                "Status": status
            })
            st.success(f"✅ Task '{name}' added!")

    # Task Table
    if st.session_state.tasks:
        st.markdown("### 📋 Current Task List")

        task_df = pd.DataFrame(st.session_state.tasks)

        # Color-code status
        def highlight_status(val):
            color = {
                "Completed": "#b6fcb6",
                "In Progress": "#fff3cd",
                "Not Started": "#f8d7da"
            }.get(val, "white")
            return f'background-color: {color}'


        # Check for overdue or due today
        today = pd.to_datetime("today").normalize()
        task_df["Deadline"] = pd.to_datetime(task_df["Deadline"])
        task_df["Due Alert"] = task_df["Deadline"].apply(
            lambda d: "🔴 Overdue" if d < today else "🟡 Due Today" if d == today else "✅ Upcoming"
        )

        st.dataframe(task_df.style.applymap(highlight_status, subset=["Status"]))

        # Task Summary Pie Chart
        st.markdown("### 📊 Task Progress Summary")
        status_counts = task_df["Status"].value_counts()
        st.pyplot(status_counts.plot.pie(autopct="%1.1f%%", ylabel="").figure)

        # Remove task
        remove_task = st.selectbox("🗑️ Select Task to Remove", options=[t["Task Name"] for t in st.session_state.tasks])
        if st.button("Remove Selected Task"):
            st.session_state.tasks = [t for t in st.session_state.tasks if t["Task Name"] != remove_task]
            st.success(f"❌ Task '{remove_task}' removed.")

    # File Upload
    st.markdown("### 📤 Upload Project Files")
    files = st.file_uploader("Choose files", accept_multiple_files=True)

    if files:
        st.session_state.uploaded_files += files
        st.success("✅ Files uploaded!")

    if st.session_state.uploaded_files:
        st.markdown("### 📂 Uploaded Files")
        for file in st.session_state.uploaded_files:
            st.write(f"📄 {file.name}")
            if file.type.startswith("image/"):
                st.image(file)
            elif file.type.startswith("text/"):
                content = file.read().decode("utf-8")
                st.text_area(f"Preview of {file.name}", content, height=150)


# TOOLKIT
elif menu == "📐 Engineering Toolkit":
    st.subheader("🧮 Engineering Calculator (Symbolic + Numeric)")
    calc_type = st.radio("Choose Mode:", ["Symbolic", "Numeric"])

    x = sp.symbols("x")  # General symbolic variable for SymPy

    if calc_type == "Symbolic":
        st.markdown("### ✳️ Symbolic Math")
        expr = st.text_input("Enter expression (e.g., x**3 + 2*x + 1)")
        operation = st.selectbox("Choose Operation", ["Differentiate", "Integrate", "Solve Equation", "Taylor Series"])

        var = st.text_input("Variable (default: x)", value="x")

        if st.button("🔍 Compute Symbolically"):
            try:
                sym_var = sp.Symbol(var)
                parsed_expr = sp.sympify(expr)

                if operation == "Differentiate":
                    result = sp.diff(parsed_expr, sym_var)
                    st.latex(f"\\frac{{d}}{{d{var}}}({sp.latex(parsed_expr)}) = {sp.latex(result)}")

                elif operation == "Integrate":
                    result = sp.integrate(parsed_expr, sym_var)
                    st.latex(f"\\int {sp.latex(parsed_expr)}\,d{var} = {sp.latex(result)} + C")

                elif operation == "Solve Equation":
                    solutions = sp.solve(parsed_expr, sym_var)
                    for i, sol in enumerate(solutions, 1):
                        st.write(f"Solution {i}: {sol}")

                elif operation == "Taylor Series":
                    order = st.slider("Select Taylor Series Order", 1, 10, 4)
                    taylor = sp.series(parsed_expr, sym_var, 0, order).removeO()
                    st.latex(f"{sp.latex(parsed_expr)} \\approx {sp.latex(taylor)}")
                    st.info("Expansion around 0 (Maclaurin Series)")

            except Exception as e:
                st.error(f"⚠️ Error: {e}")

    else:
        st.markdown("### 📈 Graph Plotting")

        f_input = st.text_area("Enter function(s) to plot (separate multiple with commas)", "sin(x), cos(x)")
        domain = st.slider("Select x-range", -50, 50, (-10, 10))
        show_values = st.checkbox("Show evaluated values", value=False)

        if st.button("📊 Plot Function(s)"):
            try:
                x_vals = np.linspace(domain[0], domain[1], 500)
                fig, ax = plt.subplots()
                func_list = [f.strip() for f in f_input.split(",")]

                for func in func_list:
                    y_vals = eval(f"np.{func}")
                    ax.plot(x_vals, y_vals, label=f"${func}$")

                    if show_values:
                        st.write(f"**Function:** `{func}`")
                        st.write(pd.DataFrame({
                            "x": np.round(x_vals, 2),
                            "y": np.round(y_vals, 4)
                        }).head(10))

                ax.set_title("Function Plot")
                ax.set_xlabel("x")
                ax.set_ylabel("f(x)")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"⚠️ Error evaluating function(s): {e}")


# GANTT CHART
elif menu == "📊 Gantt Chart":
    st.subheader("📅 Auto-Generated Gantt Chart with Progress Tracking")

    if st.session_state.tasks:
        df = pd.DataFrame(st.session_state.tasks)

        # Convert deadlines to datetime
        df['End'] = pd.to_datetime(df['deadline'])
        df['Start'] = datetime.today()

        # Add Duration column
        df['Duration (days)'] = (df['End'] - df['Start']).dt.days.clip(lower=1)

        # Map status to colors
        status_colors = {
            "Not Started": "red",
            "In Progress": "orange",
            "Completed": "green"
        }
        df['Color'] = df['status'].map(status_colors)

        # Build Gantt chart with Plotly
        fig = go.Figure()

        for i, row in df.iterrows():
            fig.add_trace(go.Bar(
                x=[row['Duration (days)']],
                y=[row['name']],
                base=row['Start'],
                orientation='h',
                marker_color=row['Color'],
                name=row['status'],
                hovertemplate=(
                    f"<b>Task:</b> {row['name']}<br>"
                    f"<b>Status:</b> {row['status']}<br>"
                    f"<b>Start:</b> {row['Start'].strftime('%Y-%m-%d')}<br>"
                    f"<b>End:</b> {row['End'].strftime('%Y-%m-%d')}<br>"
                    f"<b>Duration:</b> {row['Duration (days)']} day(s)"
                )
            ))

        fig.update_layout(
            title="📊 Project Timeline",
            xaxis_title="Date",
            yaxis_title="Task Name",
            barmode='stack',
            height=50 + len(df) * 40,
            template="plotly_white",
            showlegend=False,
            margin=dict(l=120, r=40, t=50, b=40),
        )

        fig.update_xaxes(type='date')

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ No tasks have been added yet. Please add tasks under the '📁 Project Manager' section.")

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

    if OPENAI_API_KEY is None:
        st.error("AI Assistant is unavailable due to missing API key.")
    else:
        question = st.text_area("Ask a technical engineering question:")

        if st.button("Get Answer"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Thinking... 🤖"):
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system",
                                 "content": "You are a smart engineering assistant who explains complex concepts "
                                            "clearly."},
                                {"role": "user", "content": question},
                            ],
                            max_tokens=500,
                            temperature=0.3
                        )
                        answer = response.choices[0].message.content
                        st.markdown("### Answer:")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Oops, something went wrong: {e}")
elif menu == "📈 Data Analyzer":
    st.subheader("📊 Advanced Data Analyzer & ML Toolkit")

    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown("### 🔍 Data Preview")
        st.dataframe(df.head())

        st.markdown("### 📋 Basic Info")
        st.write(f"**Shape:** {df.shape}")
        st.write("**Columns:**", df.columns.tolist())
        st.write("**Data Types:**")
        st.write(df.dtypes)

        st.markdown("### 📊 Descriptive Statistics")
        st.write(df.describe())

        st.markdown("### 🧩 Missing Data Summary")
        st.dataframe(df.isnull().sum().reset_index().rename(columns={0: "Missing Count", "index": "Column"}))

        # Column selection
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        col_x = st.selectbox("📌 Select X-axis", numeric_cols)
        col_y = st.selectbox("📌 Select Y-axis (Optional)", ["None"] + numeric_cols)

        st.markdown("### 📈 Visualization")
        plot_type = st.selectbox("Choose plot type",
                                 ["Histogram", "Box Plot", "Line Chart", "Scatter Plot", "Correlation Heatmap"])

        if plot_type == "Histogram":
            fig = px.histogram(df, x=col_x)
            st.plotly_chart(fig)
        elif plot_type == "Box Plot":
            fig = px.box(df, y=col_x)
            st.plotly_chart(fig)
        elif plot_type == "Line Chart":
            fig = px.line(df, x=df.index, y=col_x)
            st.plotly_chart(fig)
        elif plot_type == "Scatter Plot":
            if col_y != "None":
                fig = px.scatter(df, x=col_x, y=col_y)
                st.plotly_chart(fig)
            else:
                st.warning("Select Y-axis for scatter plot.")
        elif plot_type == "Correlation Heatmap":
            fig = px.imshow(df[numeric_cols].corr(), text_auto=True, color_continuous_scale="Viridis")
            st.plotly_chart(fig)

        st.markdown("### 🔎 Custom Query")
        query = st.text_input("Enter a pandas query (e.g., Age > 30 and Salary < 50000)")
        if query:
            try:
                filtered_df = df.query(query)
                st.dataframe(filtered_df)
                st.download_button("📥 Download Filtered Data", filtered_df.to_csv(index=False),
                                   file_name="filtered_data.csv")
            except Exception as e:
                st.error(f"Query Error: {e}")

        # ML SECTION
        st.markdown("### 🤖 Basic Machine Learning")
        ml_mode = st.selectbox("Select ML Task", ["None", "Regression", "Classification", "Clustering (KMeans)",
                                                  "Time Series Forecasting"])

        if ml_mode != "None":
            target = st.selectbox("🎯 Select Target Variable", df.columns)
            if ml_mode in ["Regression", "Classification"]:
                X = df.drop(columns=[target])
                y = df[target]
                X = X.select_dtypes(include=np.number).dropna()
                y = y.loc[X.index]

                if len(X) > 10:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    if ml_mode == "Regression":
                        model = LinearRegression()
                    else:
                        model = RandomForestClassifier()

                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    y_pred = model.predict(X_test)

                    st.write("**Model Accuracy / R² Score:**", score)
                    st.write("**Predictions Sample:**")
                    st.dataframe(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).head())
                else:
                    st.warning("Insufficient data for ML modeling.")

            elif ml_mode == "Clustering (KMeans)":
                X = df[numeric_cols].dropna()
                n_clusters = st.slider("Select number of clusters", 2, 10, 3)
                kmeans = KMeans(n_clusters=n_clusters)
                df['Cluster'] = kmeans.fit_predict(X)
                st.success("Clustering Completed!")
                st.dataframe(df[['Cluster'] + numeric_cols[:2]])
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color=df['Cluster'].astype(str))
                st.plotly_chart(fig)

            elif ml_mode == "Time Series Forecasting":
                date_col = st.selectbox("Select Date Column", df.columns)
                value_col = st.selectbox("Select Value Column", numeric_cols)
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(by=date_col)
                series = df.set_index(date_col)[value_col].dropna()
                try:
                    model = ARIMA()
                    results = model.fit()
                    forecast = results.forecast(steps=10)
                    st.write("**10-Step Forecast:**")
                    st.line_chart(forecast)
                except Exception as e:
                    st.error(f"ARIMA Forecasting Error: {e}")
    else:
        st.info("📁 Please upload a CSV file to begin.")
elif menu == "📋 BOM Manager":
    st.subheader("📋 Bill of Materials (BOM) Manager")

    # Initialize session state for BOM
    if "bom_data" not in st.session_state:
        st.session_state.bom_data = []

    # --- FORM FOR NEW ENTRY ---
    with st.expander("➕ Add New BOM Item"):
        with st.form("add_bom_form"):
            part_name = st.text_input("🔩 Part Name")
            part_id = st.text_input("🔖 Part ID")
            quantity = st.number_input("🔢 Quantity", min_value=1, step=1)
            unit_price = st.number_input("💰 Unit Price ($)", min_value=0.0, step=0.01, format="%.2f")
            supplier = st.text_input("🏢 Supplier")
            delivery_date = st.date_input("🚚 Expected Delivery Date")

            submitted = st.form_submit_button("Add Item")
            if submitted and part_name and part_id:
                st.session_state.bom_data.append({
                    "Part Name": part_name,
                    "Part ID": part_id,
                    "Quantity": quantity,
                    "Unit Price": unit_price,
                    "Supplier": supplier,
                    "Delivery Date": delivery_date,
                    "Total Price": quantity * unit_price
                })
                st.success("✅ Item added successfully!")

    # --- DISPLAY TABLE ---
    if st.session_state.bom_data:
        st.markdown("### 📦 Current BOM List")
        bom_df = pd.DataFrame(st.session_state.bom_data)

        # Update total price column if quantities/prices were edited
        bom_df["Total Price"] = bom_df["Quantity"] * bom_df["Unit Price"]
        total_cost = bom_df["Total Price"].sum()

        edited_df = st.data_editor(bom_df, use_container_width=True, num_rows="dynamic")
        st.session_state.bom_data = edited_df.to_dict("records")

        st.markdown(f"### 💵 **Total Estimated Cost:** ${total_cost:,.2f}")
        st.markdown(f"### 📦 **Total Unique Items:** {len(bom_df)}")

        # Export
        csv = bom_df.to_csv(index=False)
        st.download_button("📥 Download BOM as CSV", csv, file_name="bom_export.csv")

        # Optional: Visualize delivery schedule
        with st.expander("📅 Visualize Delivery Timeline (Gantt Chart)"):
            bom_df["Start"] = pd.to_datetime(datetime.today())
            bom_df["End"] = pd.to_datetime(bom_df["Delivery Date"])
            fig, ax = plt.subplots(figsize=(10, 4))
            for i, row in bom_df.iterrows():
                ax.barh(row["Part Name"], (row["End"] - row["Start"]).days, left=row["Start"], color="orange")
            ax.set_xlabel("Date")
            ax.set_ylabel("Part Name")
            st.pyplot(fig)

    else:
        st.info("No items in your BOM list yet. Add some above to get started.")

elif menu == "🧠 Calculator":
    st.subheader("🧠 Advanced Engineering Calculator")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["🧮 Symbolic Math", "🔢 Numeric Solver", "📐 Unit Conversion", "🧾 Matrix Tools", "📊 Plot Functions",
         "📚 Engineering Constants"]
    )

    # Tab 1: Symbolic Math
    with tab1:
        st.markdown("### 🧮 Symbolic Math")
        expr = st.text_input("Enter expression (e.g., x**2 + 3*x + 2)")
        var = st.text_input("Variable (e.g., x)", value="x")
        operation = st.selectbox("Select operation", ["Differentiate", "Integrate", "Simplify", "Expand"])
        if st.button("Solve Symbolically"):
            try:
                x = sp.Symbol(var)
                parsed = sp.sympify(expr)
                if operation == "Differentiate":
                    result = sp.diff(parsed, x)
                elif operation == "Integrate":
                    result = sp.integrate(parsed, x)
                elif operation == "Simplify":
                    result = sp.simplify(parsed)
                else:
                    result = sp.expand(parsed)
                st.success("Result:")
                st.latex(sp.latex(result))
            except Exception as e:
                st.error(f"Error: {e}")

    # Tab 2: Numeric Solver
    with tab2:
        st.markdown("### 🔢 Solve Equations Numerically")
        eqn = st.text_input("Enter equation (e.g., x**2 - 4 = 0)")
        variable = st.text_input("Variable", value="x")
        guess = st.number_input("Initial guess", value=1.0)
        if st.button("Solve Numerically"):
            try:
                x = sp.Symbol(variable)
                lhs, rhs = eqn.split('=')
                f_expr = sp.sympify(lhs) - sp.sympify(rhs)
                f = sp.lambdify(x, f_expr)
                from scipy.optimize import fsolve

                sol = fsolve(f, guess)
                st.success(f"Approximate solution: {sol[0]}")
            except Exception as e:
                st.error(f"Error: {e}")

    # Tab 3: Unit Conversion
    with tab3:
        st.markdown("### 📐 Unit Converter")
        from pint import UnitRegistry

        ureg = UnitRegistry()
        input_qty = st.text_input("Enter quantity with unit (e.g., 10 meter)")
        target_unit = st.text_input("Convert to (e.g., feet)")
        if st.button("Convert"):
            try:
                q = ureg
                converted = q.to(target_unit)
                st.success(f"{q} = {converted}")
            except Exception as e:
                st.error(f"Error: {e}")

    # Tab 4: Matrix Tools
    with tab4:
        st.markdown("### 🧾 Matrix Operations")
        mat_input = st.text_area("Enter matrix (comma-separated rows, e.g. '1 2; 3 4')", value="1 2; 3 4")
        operation = st.selectbox("Select matrix operation", ["Determinant", "Inverse", "Eigenvalues", "Transpose"])
        try:
            matrix = np.array([[float(num) for num in row.split()] for row in mat_input.split(';')])
            if operation == "Determinant":
                result = np.linalg.det(matrix)
            elif operation == "Inverse":
                result = np.linalg.inv(matrix)
            elif operation == "Eigenvalues":
                result = np.linalg.eigvals(matrix)
            elif operation == "Transpose":
                result = matrix.T
            st.write("Result:")
            st.write(result)
        except Exception as e:
            st.error(f"Error: {e}")

    # Tab 5: Function Plotting
    with tab5:
        st.markdown("### 📊 Function Plotter")
        func = st.text_input("Enter function to plot (e.g., sin(x) * x)")
        domain = st.slider("Domain range for x", -50, 50, (-10, 10))
        if st.button("Plot Graph"):
            try:
                x_vals = np.linspace(domain[0], domain[1], 400)
                y_vals = eval(f"np.{func}")
                plt.figure(figsize=(8, 4))
                plt.plot(x_vals, y_vals, label=func, color='green')
                plt.title("Function Plot")
                plt.xlabel("x")
                plt.ylabel("f(x)")
                plt.grid(True)
                plt.legend()
                st.pyplot(plt.gcf())
            except Exception as e:
                st.error(f"Plotting error: {e}")

    # Tab 6: Engineering Constants
    with tab6:
        st.markdown("### 📚 Common Engineering Constants")
        constants = {
            "Gravitational Acceleration (g)": "9.81 m/s²",
            "Speed of Light (c)": "3 x 10⁸ m/s",
            "Planck’s Constant (h)": "6.626 x 10⁻³⁴ Js",
            "Boltzmann Constant (k)": "1.38 x 10⁻²³ J/K",
            "Gas Constant (R)": "8.314 J/(mol·K)",
            "Avogadro's Number": "6.022 x 10²³ 1/mol"
        }
        for key, val in constants.items():
            st.markdown(f"**{key}**: {val}")

elif menu == "📘 Engineering Glossary":
    st.subheader("📘 Engineering Glossary & Reference Hub")
    st.markdown("Search and explore key terms across major engineering disciplines.")

    # Glossary categories
    categories = {
        "Mechanical Engineering": ["Stress", "Strain", "Thermodynamics", "Torque", "Viscosity"],
        "Electrical Engineering": ["Ohm's Law", "Capacitance", "Inductance", "Current", "Voltage"],
        "Civil Engineering": ["Reinforced Concrete", "Load Bearing", "Shear Force", "Bending Moment"],
        "Chemical Engineering": ["Catalysis", "Distillation", "Reaction Rate", "Molarity"],
        "Aerospace Engineering": ["Thrust", "Drag", "Lift", "Mach Number", "Reynolds Number"],
        "Computer Engineering": ["Boolean Algebra", "Flip-Flop", "Cache Memory", "Algorithm"],
        "Data Science": ["Regression", "Clustering", "Overfitting", "Bias-Variance Tradeoff"]
    }

    # Sidebar filtering
    selected_branch = st.selectbox("Select Discipline", list(categories.keys()))
    search_term = st.text_input("🔍 Search for a term (e.g., Torque, Lift)")

    glossary = {
        "Stress": {
            "definition": "Stress is the force applied per unit area within materials.",
            "formula": "σ = F / A",
            "unit": "Pascal (Pa)",
            "field": "Mechanical Engineering"
        },
        "Ohm's Law": {
            "definition": "Defines the relationship between voltage, current, and resistance.",
            "formula": "V = IR",
            "unit": "Volts (V)",
            "field": "Electrical Engineering"
        },
        "Thrust": {
            "definition": "A reaction force that propels a body forward in space or air.",
            "formula": "T = ṁ * (v_e - v_0)",
            "unit": "Newtons (N)",
            "field": "Aerospace Engineering"
        },
        "Lift": {
            "definition": "An aerodynamic force that holds an aircraft in the air.",
            "formula": "L = 0.5 * ρ * v^2 * S * Cl",
            "unit": "Newtons (N)",
            "field": "Aerospace Engineering"
        },
        "Regression": {
            "definition": "A method to model and analyze relationships between variables.",
            "formula": "y = mx + b (simple linear regression)",
            "unit": "N/A",
            "field": "Data Science"
        },
        # Add more terms here as needed...
    }

    # Filter and show terms
    filtered_terms = [term for term in categories[selected_branch] if search_term.lower() in term.lower()]
    for term in filtered_terms:
        if term in glossary:
            st.markdown(f"### 🔹 {term}")
            st.write(f"**Definition:** {glossary[term]['definition']}")
            st.latex(glossary[term]['formula'] if glossary[term]['formula'] != "N/A" else "")
            st.write(f"**Unit:** {glossary[term]['unit']}")
            st.write(f"**Discipline:** {glossary[term]['field']}")

    if not filtered_terms and search_term:
        st.warning("No matching term found. Try a different keyword.")

elif menu == "📎 Notes & Docs":
    st.subheader("📎 Notes & Document Center")
    st.markdown("""
    Manage, create, annotate, and download engineering notes and documents.
    Upload lecture files or type your own, then visualize or export your notes!
    """)

    uploaded_file = st.file_uploader("📤 Upload file (PDF, TXT, DOCX, CSV)", type=["pdf", "txt", "docx", "csv"])

    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1]

        if file_type == "pdf":
            # Render PDF
            base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

        elif file_type == "txt":
            content = uploaded_file.read().decode("utf-8")
            st.text_area("📄 File Content", content, height=300)

        elif file_type == "csv":
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
            st.line_chart(df.select_dtypes(include=['float', 'int']))  # Visualize numerics

    st.divider()

    # 📚 CREATE NOTES FROM SCRATCH
    st.subheader("🧾 Create & Save Engineering Notes")
    title = st.text_input("Title of Notes")
    notes = st.text_area("Write your notes below:", height=300)
    file_format = st.selectbox("Choose export format:", ["TXT", "PDF", "Markdown"])

    if st.button("📥 Export Notes"):
        if notes and title:
            if file_format == "TXT":
                st.download_button("Download .txt", notes, file_name=f"{title}.txt")
            elif file_format == "Markdown":
                st.download_button("Download .md", notes, file_name=f"{title}.md")
            elif file_format == "PDF":
                # Generate a temporary PDF
                from fpdf import FPDF

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                for line in notes.split('\n'):
                    pdf.cell(200, 10, txt=line, ln=True)
                pdf_path = f"/tmp/{title}.pdf"
                pdf.output(pdf_path)
                with open(pdf_path, "rb") as file:
                    st.download_button("Download .pdf", file.read(), file_name=f"{title}.pdf", mime="application/pdf")
        else:
            st.warning("Title and content required.")

    st.divider()

    # 📌 AI SUMMARY (optional if user prefers)
    st.subheader("🧠 Quick Summary of Your Notes")
    if notes:
        with st.spinner("Summarizing your notes..."):

            try:
                summary = summarize(notes)
                st.success("📝 Summary:")
                st.markdown(summary)
            except:
                st.error("Summary failed — text may be too short or complex.")

    # 📊 Visualization of keywords
    if notes:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        wordcloud = WordCloud(background_color='white', width=800, height=300).generate(notes)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.subheader("🔍 WordCloud of Your Notes")
        st.pyplot(fig)

elif menu == "🔁 Version Control":
    st.subheader("🔁 Version Control Center")

    # Initialize session state
    if "versions" not in st.session_state:
        st.session_state.versions = []

    # Uploading new version
    st.markdown("### 📥 Upload New Version")
    uploaded_file = st.file_uploader("Upload a file (PDF, DOCX, TXT, CSV)", type=["pdf", "docx", "txt", "csv"])
    version_note = st.text_input("Version Note (e.g., Initial Draft, Final Version v1.2)")

    if st.button("Save Version"):
        if uploaded_file and version_note:
            st.session_state.versions.append({
                "filename": uploaded_file.name,
                "data": uploaded_file.read(),
                "note": version_note,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success("✅ Version saved successfully!")
        else:
            st.warning("⚠️ Please upload a file and add a version note.")

    # Show all saved versions
    if st.session_state.versions:
        st.markdown("### 🗂 Version History")
        for idx, version in enumerate(reversed(st.session_state.versions), 1):
            with st.expander(f"📄 Version {len(st.session_state.versions) - idx + 1} - {version['note']}"):
                st.write(f"🕒 Timestamp: `{version['timestamp']}`")
                st.write(f"📎 File: `{version['filename']}`")
                if version['filename'].endswith(".txt"):
                    content = version["data"].decode("utf-8")
                    st.code(content, language='text')
                elif version['filename'].endswith(".csv"):
                    df = pd.read_csv(io.BytesIO(version["data"]))
                    st.dataframe(df)
                elif version['filename'].endswith(".pdf"):
                    base64_pdf = base64.b64encode(version["data"]).decode("utf-8")
                    st.markdown(
                        f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400px"></iframe>',
                        unsafe_allow_html=True)

    # Optional: Compare Two Versions
    if len(st.session_state.versions) >= 2:
        st.markdown("### 🔍 Compare Two Versions")
        file_options = [f"{i + 1}: {v['note']}" for i, v in enumerate(st.session_state.versions)]
        version1 = st.selectbox("Select First Version", options=file_options, index=0)
        version2 = st.selectbox("Select Second Version", options=file_options, index=1)

        idx1 = int(version1.split(":")[0]) - 1
        idx2 = int(version2.split(":")[0]) - 1

        data1 = st.session_state.versions[idx1]["data"]
        data2 = st.session_state.versions[idx2]["data"]

        try:
            diff1 = data1.decode()
            diff2 = data2.decode()
            from difflib import ndiff

            diff_result = '\n'.join(ndiff(diff1.splitlines(), diff2.splitlines()))
            st.code(diff_result, language="diff")
        except:
            st.warning("⚠️ Cannot compare these files. Only text-based file comparisons are supported.")

elif menu == "🛠️ Unit Converter":
    st.subheader("🛠️ Engineering Unit Converter")

    categories = {
        "Length": {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "ft": 0.3048},
        "Force": {"N": 1.0, "kN": 1000, "lbf": 4.44822},
        "Temperature": {"°C": "celsius", "°F": "fahrenheit", "K": "kelvin"},
        "Pressure": {"Pa": 1.0, "kPa": 1000, "MPa": 1e6, "psi": 6894.76, "atm": 101325}
    }

    category = st.selectbox("Choose category", list(categories.keys()))

    if category != "Temperature":
        units = list(categories[category].keys())
        input_val = st.number_input("Enter value", format="%0.5f")
        from_unit = st.selectbox("From", units)
        to_unit = st.selectbox("To", units)
        if st.button("Convert"):
            result = input_val * categories[category][from_unit] / categories[category][to_unit]
            st.success(f"{input_val} {from_unit} = {result:.5f} {to_unit}")
    else:
        temp_input = st.number_input("Enter temperature")
        from_unit = st.selectbox("From", ["°C", "°F", "K"])
        to_unit = st.selectbox("To", ["°C", "°F", "K"])
        def convert_temp(value, from_u, to_u):
            if from_u == to_u: return value
            if from_u == "°C":
                if to_u == "°F": return value * 9/5 + 32
                if to_u == "K": return value + 273.15
            elif from_u == "°F":
                if to_u == "°C": return (value - 32) * 5/9
                if to_u == "K": return (value - 32) * 5/9 + 273.15
            elif from_u == "K":
                if to_u == "°C": return value - 273.15
                if to_u == "°F": return (value - 273.15) * 9/5 + 32
        if st.button("Convert Temperature"):
            converted = convert_temp(temp_input, from_unit, to_unit)
            st.success(f"{temp_input} {from_unit} = {converted:.2f} {to_unit}")

elif menu == "📦 Inventory Tracker":
    st.subheader("📦 Engineering Inventory Tracker")

    if 'inventory' not in st.session_state:
        st.session_state.inventory = []

    with st.form("add_inventory"):
        name = st.text_input("Item Name")
        quantity = st.number_input("Quantity", min_value=0, step=1)
        location = st.text_input("Location (e.g., Store Room A)")
        category = st.selectbox("Category", ["Mechanical", "Electrical", "Thermal", "Other"])
        add_item = st.form_submit_button("Add Item")
        if add_item and name:
            st.session_state.inventory.append({
                "Item": name, "Quantity": quantity, "Location": location,
                "Category": category, "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success("Item added!")

    if st.session_state.inventory:
        df = pd.DataFrame(st.session_state.inventory)
        st.dataframe(df)

        st.markdown("### 🔎 Filter/Search Inventory")
        search_term = st.text_input("Search by name/category:")
        if search_term:
            filtered = df[df.apply(lambda row: search_term.lower() in row.astype(str).str.lower().values, axis=1)]
            st.dataframe(filtered)

        if st.button("📤 Export Inventory to CSV"):
            csv = df.to_csv(index=False).encode()
            st.download_button("Download CSV", csv, "inventory_export.csv", "text/csv")
elif menu == "🎯 Goal Tracker":
    st.subheader("🎯 Engineering Goal Tracker")

    if "goals" not in st.session_state:
        st.session_state.goals = []

    with st.form("add_goal_form"):
        goal = st.text_input("Enter your goal")
        deadline = st.date_input("Deadline", datetime.date.today())
        add_goal = st.form_submit_button("Add Goal")
        if add_goal and goal:
            st.session_state.goals.append({
                "Goal": goal,
                "Deadline": deadline.strftime("%Y-%m-%d"),
                "Status": "Pending"
            })
            st.success("Goal added successfully!")

    if st.session_state.goals:
        df_goals = pd.DataFrame(st.session_state.goals)
        st.dataframe(df_goals)

        completed = st.multiselect("Mark Completed Goals", df_goals["Goal"])
        for i, g in enumerate(st.session_state.goals):
            if g["Goal"] in completed:
                st.session_state.goals[i]["Status"] = "Completed"
elif menu == "🧪 Test Log":
    st.subheader("🧪 Engineering Test Logbook")

    if "test_logs" not in st.session_state:
        st.session_state.test_logs = []

    with st.form("test_log_form"):
        test_name = st.text_input("Test Name")
        date = st.date_input("Test Date")
        outcome = st.selectbox("Outcome", ["Passed", "Failed", "Pending"])
        remarks = st.text_area("Remarks")
        if st.form_submit_button("Log Test"):
            st.session_state.test_logs.append({
                "Test Name": test_name,
                "Date": date.strftime("%Y-%m-%d"),
                "Outcome": outcome,
                "Remarks": remarks
            })
            st.success("Test logged.")

    if st.session_state.test_logs:
        df_tests = pd.DataFrame(st.session_state.test_logs)
        st.dataframe(df_tests)
elif menu == "🕒 Time Tracker":
    st.subheader("🕒 Work Session Time Tracker")

    if "start_time" not in st.session_state:
        st.session_state.start_time = None
        st.session_state.time_log = []

    if st.button("▶️ Start Timer"):
        st.session_state.start_time = datetime.datetime.now()
        st.success("Timer started!")

    if st.button("⏹️ Stop Timer") and st.session_state.start_time:
        end_time = datetime.datetime.now()
        duration = end_time - st.session_state.start_time
        st.session_state.time_log.append({
            "Start": st.session_state.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "End": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Duration": str(duration)
        })
        st.session_state.start_time = None
        st.success(f"Session logged: {duration}")

    if st.session_state.time_log:
        df_time = pd.DataFrame(st.session_state.time_log)
        st.dataframe(df_time)
elif menu == "🔍 Error Log Analyzer":
    st.subheader("🔍 Log Analyzer")

    log_file = st.file_uploader("Upload Error Log (.txt)", type=["txt"])
    if log_file:
        content = log_file.read().decode("utf-8")
        st.text_area("Log Content", content, height=300)
        import re
        errors = re.findall(r"(?i)error:?.*", content)
        st.markdown("### ⚠️ Extracted Errors")
        for err in errors:
            st.error(err)
elif menu == "📡 Sensor Monitor":
    st.subheader("📡 Real-Time Sensor Data Monitor (Simulated)")

    st.markdown("📈 Simulated output for temperature, voltage, and RPM")

    t = np.arange(0, 10, 0.1)
    temp = 25 + 5*np.sin(t)
    voltage = 12 + np.cos(t)
    rpm = 1500 + 300*np.sin(2*t)

    fig, ax = plt.subplots()
    ax.plot(t, temp, label="Temp (°C)")
    ax.plot(t, voltage, label="Voltage (V)")
    ax.plot(t, rpm, label="RPM")
    ax.set_title("Sensor Readings")
    ax.legend()
    st.pyplot(fig)
elif menu == "🧰 Tool Scheduler":
    st.subheader("🧰 Schedule Tool Usage")

    if "tools" not in st.session_state:
        st.session_state.tools = []

    with st.form("tool_form"):
        tool = st.text_input("Tool Name")
        user = st.text_input("User")
        date = st.date_input("Usage Date")
        time = st.time_input("Usage Time")
        schedule = st.form_submit_button("Schedule Tool")
        if schedule:
            st.session_state.tools.append({
                "Tool": tool,
                "User": user,
                "Scheduled": f"{date} {time}"
            })
            st.success("Scheduled!")

    if st.session_state.tools:
        df_tool = pd.DataFrame(st.session_state.tools)
        st.dataframe(df_tool)
elif menu == "🎵📏 Sound metre pro":
    st.subheader("🎵📏 Sound Level Simulation")

    db_level = st.slider("🔊 Simulated Sound Level (dB)", 20, 120, 70)
    if db_level > 85:
        st.warning("⚠️ Dangerous sound level!")
    elif db_level > 60:
        st.info("👂 Moderate noise")
    else:
        st.success("🟢 Safe level")

elif menu == "🎥🔧 Engineering videos":
    st.subheader("🎥🔧 Watch Engineering Tutorials")

    st.video("https://www.youtube.com/watch?v=W74y1RxN6BA&pp=ygUXbWVjaGFuaWNhbCBlbmdpbmVlcmluZyA%3D")
    st.video("https://www.youtube.com/watch?v=dS4VU8KchWQ&pp=ygUWZWxlY3RyaWNhbCBlbmdpbmVlcmluZw%3D%3D")
    st.video("https://www.youtube.com/watch?v=bFljMHTQ1QY&pp=ygURY2l2aWwgZW5naW5lZXJpbmc%3D")

st.markdown("---")
st.markdown("""
  <center>
      🔧 Developed by **Praise Adeyeye**  
      🧠 _"Engineering isn’t just equations — it’s imagination made real."_ 💡  
      <br>
      © 2025 Praise Adeyeye. All rights reserved. 🚀
  </center>
  """, unsafe_allow_html=True)
