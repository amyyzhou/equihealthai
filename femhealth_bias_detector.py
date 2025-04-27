import streamlit as st
import pandas as pd
import openai
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score


# Set OpenAI API Key

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def ask_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a medical AI fairness advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

st.image("/Users/amyzhou/Downloads/Frame_10-removebg-preview.png", width=150)
st.title("EquiHealth AI")
st.write(
    "A tool to help evaluate and mitigate bias in healthcare machine learning models by applying fairness techniques."
)

uploaded_file = st.file_uploader("Upload your healthcare dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset", df.head())

    target_column = st.selectbox("Select the target outcome column", df.columns)
    sensitive_column = st.selectbox("Select the sensitive attribute (e.g., gender)", df.columns)
    feature_columns = st.multiselect("Select feature columns for training", df.columns.drop([target_column]))

    suggested_categoricals = [col for col in feature_columns if df[col].dtype == 'object' or df[col].nunique() <= 10]
    categorical_columns = st.multiselect("Categorical Columns (Auto-suggested):", feature_columns, default=suggested_categoricals)

    # Ensure binary target
    if df[target_column].nunique() > 2:
        df[target_column] = df[target_column].apply(lambda x: 0 if x == 0 else 1)

    # Add this before SMOTE section:
    run_baseline = st.checkbox("Run Without Fairness Techniques or SMOTE (Baseline Model)")

    if run_baseline:
        apply_smote = False
        apply_inprocessing = False
        apply_postprocessing = False
        train_separately = False
    else:

        # SMOTE Logic
        apply_smote = False
        target_counts = df[target_column].value_counts(normalize=True)
        minority_pct = target_counts.min()

        if minority_pct < 0.35:
            st.warning(f"Class imbalance detected ({round(minority_pct*100,1)}%). SMOTE recommended.")
            apply_smote = st.checkbox("Apply SMOTE", value=True)
        elif minority_pct < 0.45:
            st.info(f"Slight imbalance ({round(minority_pct*100,1)}%). SMOTE optional.")
            apply_smote = st.checkbox("Apply SMOTE?", value=False)
        else:
            st.success(f"Classes well balanced ({round(minority_pct*100,1)}%). No SMOTE needed.")

        # Clinical Context & Fairness
        # Initialize variables
        apply_inprocessing = st.checkbox("Apply In-Processing Fairness (ExponentiatedGradient)")

        fairness_strategy = None
        train_separately = False
        apply_postprocessing = False

        if apply_inprocessing:
            # Now ask about biological differences ONLY if In-Processing is selected
            clinical_diff = st.radio("Are there known biological differences across genders?", ["Yes", "No", "Not Sure"])

            if clinical_diff == "Yes":
                st.info("Recommendation: Use Equalized Odds or train gender-specific models.")
                fairness_choice = st.selectbox("Select Fairness Approach:", ["Equalized Odds", "Train Separate Models"])
                if fairness_choice == "Train Separate Models":
                    train_separately = True
                else:
                    fairness_strategy = EqualizedOdds()

            elif clinical_diff == "No":
                st.info("Defaulting to Demographic Parity.")
                fairness_strategy = DemographicParity()

            else:
                st.info("Defaulting to Equalized Odds.")
                fairness_strategy = EqualizedOdds()

        # Post-Processing toggle (only if not training separately)
        if not train_separately:
            apply_postprocessing = st.checkbox("Apply Post-Processing Threshold Optimization")

        # GPT Explanation Button
        if st.button("Why this fairness recommendation?"):
            if clinical_diff == "Yes" and apply_inprocessing:
                selected_strategy = fairness_choice
            else:
                selected_strategy = fairness_strategy.__class__.__name__ if fairness_strategy else "No Fairness Applied"
            
            gpt_prompt = f"For {uploaded_file.name}, why is {selected_strategy} recommended given biological differences = {clinical_diff}?"
            st.chat_message("assistant").write(ask_gpt(gpt_prompt))


        # Run Bias Audit
    if st.button("Run Bias Audit"):
        imputer = SimpleImputer(strategy='most_frequent')  # Create ONE imputer instance

        if train_separately:
            genders = df[sensitive_column].unique()
            for gender in genders:
                df_gender = df[df[sensitive_column] == gender]
                X = df_gender[feature_columns]
                y = df_gender[target_column]
                X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Use single imputer instance
                X_train = imputer.fit_transform(X_train)
                X_test = imputer.transform(X_test)

                if apply_smote:
                    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

                model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.text(f"Performance for {gender}:")
                st.text(classification_report(y_test, y_pred))

        else:
            X = pd.get_dummies(df[feature_columns], columns=categorical_columns, drop_first=True)
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Use single imputer instance
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

            if apply_smote:
                X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

            base_model = RandomForestClassifier(random_state=42)
            if apply_inprocessing and fairness_strategy:
                model = ExponentiatedGradient(base_model, constraints=fairness_strategy)
                model.fit(X_train, y_train, sensitive_features=df.loc[y_train.index, sensitive_column])
            else:
                model = base_model.fit(X_train, y_train)

            if apply_postprocessing:
                if hasattr(model, "predict_proba"):
                    thresh_opt = ThresholdOptimizer(estimator=model, constraints="equalized_odds", prefit=True, predict_method='predict_proba')
                    thresh_opt.fit(X_train, y_train, sensitive_features=df.loc[y_train.index, sensitive_column])
                    y_pred = thresh_opt.predict(X_test, sensitive_features=df.loc[y_test.index, sensitive_column])
                else:
                    st.error("Post-Processing requires a model that supports probability predictions. Cannot apply ThresholdOptimizer after In-Processing with this setup.")
                    y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)

            st.text("Model Performance:")
            st.text(classification_report(y_test, y_pred))

            # Assuming sensitive_column holds gender info
            sensitive_features_test = df.loc[y_test.index, sensitive_column]

            # Use MetricFrame to disaggregate performance
            metric_frame = MetricFrame(
                metrics={"accuracy": accuracy_score},
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=sensitive_features_test
            )

            # Calculate fairness metrics
            dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features_test)
            eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_features_test)

            # Display in Streamlit
            st.write("### Fairness Metrics:")
            st.write(f"Demographic Parity Difference: `{dp_diff:.3f}`")
            st.write(f"Equalized Odds Difference: `{eo_diff:.3f}`")

            # Optional: Show accuracy per group
            st.write("### Accuracy by Group:")
            st.dataframe(metric_frame.by_group)

    # --- AI Question Section ---
    if st.button("Ask AI About Results"):
        result_question = st.text_input("Enter your question about model fairness or performance:")
        if result_question:
            st.chat_message("assistant").write(ask_gpt(f"User question: {result_question}"))
