import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer

# 1. Upload Dataset
st.title("FemHealth Bias Detector")
uploaded_file = st.file_uploader("Upload your healthcare dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset", df.head())

    # 2. User selects columns
    target_column = st.selectbox("Select the target outcome column", df.columns)
    sensitive_column = st.selectbox("Select the sensitive (e.g., gender) column", df.columns)

    feature_columns = st.multiselect("Select feature columns for training", df.columns.drop([target_column]))

    # 3. Auto-suggest categorical columns
    suggested_categoricals = [col for col in feature_columns if df[col].dtype == 'object' or df[col].nunique() <= 10]

    categorical_columns = st.multiselect(
        "Categorical Columns (Auto-suggested below, adjust if needed):",
        feature_columns,
        default=suggested_categoricals
    )

    # 4. Offer Binary Conversion if Target is Multi-class
    binary_conversion = False
    if df[target_column].nunique() > 2:
        if st.checkbox(f"Convert '{target_column}' to binary classification (0 = No disease, 1 = Disease present)"):
            df[target_column] = df[target_column].apply(lambda x: 0 if x == 0 else 1)
            st.write("✅ Target column converted to binary.")
            binary_conversion = True

    # 5. Auto-suggest resampling if imbalance detected
    if sensitive_column in df.columns:
        group_counts = df[sensitive_column].value_counts(normalize=True)
        if group_counts.min() < 0.4:
            st.warning("⚠️ Detected imbalance in the sensitive feature. SMOTE is recommended.")
            apply_smote = st.checkbox("Apply SMOTE to balance the training set", value=True)
        else:
            apply_smote = False
    else:
        apply_smote = False

    # 6. In-processing and Post-processing options
    apply_inprocessing = st.checkbox("Apply In-Processing Fairness Constraint (Fairlearn)")
    apply_postprocessing = st.checkbox("Apply Post-Processing Threshold Optimization")

    if st.button("Run Bias Audit"):
        # Prepare features and target
        X = df[feature_columns]
        y = df[target_column]

        # Validate target for fairness methods
        if (apply_inprocessing or apply_postprocessing):
            if y.nunique() > 2 or sorted(y.unique()) != [0, 1]:
                st.error("❌ Fairness methods require binary classification with labels 0 and 1. Please convert your target variable.")
                st.stop()

        # Apply One-Hot Encoding
        X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

        # Align columns
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

        # Apply Imputation
        imputer = SimpleImputer(strategy='most_frequent')
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

        # Apply SMOTE (Pre-processing fairness)
        if apply_smote:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        # Apply In-processing (Fairlearn)
        if apply_inprocessing:
            constraint = DemographicParity()
            base_model = RandomForestClassifier(random_state=42)
            model = ExponentiatedGradient(base_model, constraints=constraint)
            model.fit(X_train, y_train, sensitive_features=df.loc[y_train.index, sensitive_column])
        else:
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

        # Apply Post-processing (Threshold Optimization)
        if apply_postprocessing:
            thresh_opt = ThresholdOptimizer(
                estimator=model,
                constraints="equalized_odds",
                prefit=True,
                predict_method='predict_proba'
            )
            thresh_opt.fit(X_train, y_train, sensitive_features=df.loc[y_train.index, sensitive_column])
            y_pred = thresh_opt.predict(X_test, sensitive_features=df.loc[y_test.index, sensitive_column])
        else:
            y_pred = model.predict(X_test)

        # Evaluate Model
        st.text("Model Performance on Test Set:")
        st.text(classification_report(y_test, y_pred))