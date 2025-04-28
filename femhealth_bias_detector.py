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
import os

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_gpt(prompt):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a medical AI fairness advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

st.image("assets/Frame_10-removebg-preview.png")
st.title("EquiHealth AI")
st.write("A tool to help evaluate and mitigate bias in healthcare machine learning models by applying fairness techniques.")

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

    run_baseline = st.checkbox("Run Without Fairness Techniques or SMOTE (Baseline Model)")

    if run_baseline:
        apply_smote = False
        apply_inprocessing = False
        apply_postprocessing = False
        train_separately = False
    else:
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

        apply_inprocessing = st.checkbox("Apply In-Processing Fairness (ExponentiatedGradient)")
        fairness_strategy = None
        train_separately = False
        apply_postprocessing = False

        if apply_inprocessing:
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

        if not train_separately:
            apply_postprocessing = st.checkbox("Apply Post-Processing Threshold Optimization")

        if st.button("Why this fairness recommendation?"):
            if clinical_diff == "Yes" and apply_inprocessing:
                selected_strategy = fairness_choice
            else:
                selected_strategy = fairness_strategy.__class__.__name__ if fairness_strategy else "No Fairness Applied"
            gpt_prompt = f"For {uploaded_file.name}, why is {selected_strategy} recommended given biological differences = {clinical_diff}?"
            st.chat_message("assistant").write(ask_gpt(gpt_prompt))

    if st.button("Run Bias Audit"):
        imputer = SimpleImputer(strategy='most_frequent')

        if train_separately:
            genders = df[sensitive_column].unique()
            for gender in genders:
                df_gender = df[df[sensitive_column] == gender]
                X = pd.get_dummies(df_gender[feature_columns], columns=categorical_columns, drop_first=True)
                y = df_gender[target_column].astype(int)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
                X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

                if apply_smote:
                    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

                model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.text(f"Performance for {gender}:")
                st.text(classification_report(y_test, y_pred))

        else:
            X = pd.get_dummies(df[feature_columns], columns=categorical_columns, drop_first=True)
            y = df[target_column].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

            if apply_smote:
                X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

            base_model = RandomForestClassifier(random_state=42)
            if apply_inprocessing and fairness_strategy:
                model = ExponentiatedGradient(base_model, constraints=fairness_strategy)
                model.fit(X_train, y_train, sensitive_features=df.loc[y_train.index, sensitive_column])
            else:
                # Ensure X_train is a DataFrame and all columns are numeric
                X_train = pd.DataFrame(X_train)
                X_train = X_train.apply(pd.to_numeric, errors='coerce')
            
                # Ensure y_train is numeric integers
                y_train = pd.Series(y_train).astype(int)
            
                model = base_model.fit(X_train, y_train)

            if apply_postprocessing:
                if hasattr(model, "predict_proba"):
                    thresh_opt = ThresholdOptimizer(estimator=model, constraints="equalized_odds", prefit=True, predict_method='predict_proba')
                    thresh_opt.fit(X_train, y_train, sensitive_features=df.loc[y_train.index, sensitive_column])
                    y_pred = thresh_opt.predict(X_test, sensitive_features=df.loc[y_test.index, sensitive_column])
                else:
                    st.error("Post-Processing requires probability outputs.")
                    y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)

            st.text("Model Performance:")
            st.text(classification_report(y_test, y_pred))

            sensitive_features_test = df.loc[y_test.index, sensitive_column]

            metric_frame = MetricFrame(
                metrics={"accuracy": accuracy_score},
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=sensitive_features_test
            )

            dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features_test)
            eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_features_test)

            st.write("### Fairness Metrics:")
            st.write(f"Demographic Parity Difference: `{dp_diff:.3f}`")
            st.write(f"Equalized Odds Difference: `{eo_diff:.3f}`")
            st.write("### Accuracy by Group:")
            st.dataframe(metric_frame.by_group)

    if st.button("Ask AI About Results"):
        result_question = st.text_input("Enter your question about model fairness or performance:")
        if result_question:
            st.chat_message("assistant").write(ask_gpt(f"User question: {result_question}"))
