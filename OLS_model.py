import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import re
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.api import anova_lm

# === Part 1: Model Fitting ===

# --- Configuration ---
filepath = 'Final_dataset.csv'
topic_cols = ['topic_1', 'topic_2', 'topic_3']
emotion_col = 'predicted_emotion_2'
dependent_var = 'dv_response_mean'
value_to_ignore = 'unknown'
min_freq_topic = 30
min_freq_emotion = 30
baseline_emotion = 'neutral'
emotion_dummy_prefix = 'emo'

demographic_cols_config = {
    'age': 'age',
    'gender': 'gender',
    'political': 'ideological_affiliation',
    'issue_stance': 'issue_stance'
}
demographic_predictors_prefix = 'demog'
issue_stance_prefix = 'stance'
baseline_issue_stance = 'Digital privacy'

issue_stance_mapping_cleaned_keys = {
    "the u.s. should impose stronger economic sanctions on china.": "China_sanctions",
    "the u.s. should increase investments in renewable energy technologies.": "Renewable_energy",
    "the u.s. should not implement legislation that strengthens digital privacy rights": "Digital_privacy",
    "the u.s. should not increase its support for nato.": "NATO_support"
}

def format_topic_name_for_script(topic_name):
    """Formats a topic name string to be a valid Python identifier."""
    suffix = re.sub(r'[&/ \-]', '_', topic_name)
    suffix = re.sub(r'_+', '_', suffix)
    return f"topic_{suffix}"

# --- Define Specific Interaction Terms ---
# Based on a pre-defined list of top topics for each issue stance,
# this section dynamically builds the list of interaction terms to be created in the model.
top3_topics_per_stance = {
    "China_sanctions": ["Fairness, Rights & Justice", "Security & Threats", "Economy & Jobs"],
    "Digital_privacy": ["Security & Threats", "Innovation & Technology", "Balance & Compromise"],
    "NATO_support": ["Domestic Focus", "Diplomacy & International", "Security & Threats"],
    "Renewable_energy": ["Economy & Jobs", "Environment & Sustainability", "Economy & Jobs"]
}

specific_interaction_terms_to_create = []
for stance_key, topic_list in top3_topics_per_stance.items():
    stance_dummy = f"{issue_stance_prefix}_{stance_key}"
    for topic_name in topic_list:
        topic_dummy = format_topic_name_for_script(topic_name)
        specific_interaction_terms_to_create.append((stance_dummy, topic_dummy))

# --- Load and Clean Data ---
try:
    df = pd.read_csv(filepath)
except FileNotFoundError:
    print(f"Error: File not found at {filepath}"); exit()
except Exception as e:
    print(f"An error occurred loading file: {e}"); exit()

df_processed = df.copy()
if dependent_var in df_processed.columns:
    df_processed[dependent_var] = pd.to_numeric(df_processed[dependent_var], errors='coerce')
    df_processed.dropna(subset=[dependent_var], inplace=True)
else:
    print(f"Error: Dependent variable '{dependent_var}' not found."); exit()

if emotion_col in df_processed.columns:
    df_processed[emotion_col] = df_processed[emotion_col].astype(str).str.strip().str.lower()
    df_processed = df_processed[df_processed[emotion_col] != value_to_ignore]

for key, col_name in demographic_cols_config.items():
    if col_name in df_processed.columns:
        df_processed[col_name] = df_processed[col_name].astype(str).str.strip().str.lower()
        df_processed.loc[df_processed[col_name] == value_to_ignore, col_name] = np.nan
if len(df_processed) == 0: print("Error: No data remaining after initial cleaning."); exit()


# --- Filter by Frequent Topics and Emotions ---
existing_topic_cols = [col for col in topic_cols if col in df_processed.columns]
if not existing_topic_cols: print(f"Error: No topic_cols found."); exit()
all_topics_series_initial = df_processed[existing_topic_cols].stack().dropna().astype(str).str.strip()
all_topics_series_initial = all_topics_series_initial[all_topics_series_initial != '']
all_topics_series_initial = all_topics_series_initial[all_topics_series_initial.str.lower() != value_to_ignore]
topic_overall_counts_initial = all_topics_series_initial.value_counts()
frequent_topics_orig_names = topic_overall_counts_initial[topic_overall_counts_initial >= min_freq_topic].index.tolist()
print(f"Frequent topics identified from data: {frequent_topics_orig_names}")

df_filtered = df_processed.copy()
emotion_counts_in_filtered = None
if emotion_col in df_filtered.columns:
    emotion_counts_initial = df_filtered[emotion_col].value_counts()
    frequent_emotions_values = emotion_counts_initial[emotion_counts_initial >= min_freq_emotion].index.tolist()
    df_filtered = df_filtered[df_filtered[emotion_col].isin(frequent_emotions_values)]
    emotion_counts_in_filtered = df_filtered[emotion_col].value_counts()
    if len(df_filtered) == 0: print("Error: No data remaining after emotion filtering."); exit()

# --- Process Demographics and Create Dummy Variables ---
# This block processes each demographic variable (age, gender, politics, issue stance),
# groups them into categories, and creates dummy variables for the model,
# dropping a pre-defined baseline category for each.
demographic_dummy_cols_generated = []

age_col_name = demographic_cols_config.get('age')
if age_col_name and age_col_name in df_filtered.columns:
    df_filtered['age_numeric'] = pd.to_numeric(df_filtered[age_col_name], errors='coerce')
    age_bins = [0, 29, 49, np.inf]
    age_group_labels_ordered = [f'{demographic_predictors_prefix}_age_group_young',
                                f'{demographic_predictors_prefix}_age_group_middle_aged',
                                f'{demographic_predictors_prefix}_age_group_older']
    df_filtered['age_group_cat'] = pd.cut(df_filtered['age_numeric'], bins=age_bins, labels=age_group_labels_ordered, right=True, ordered=False)
    age_group_dummies_all = pd.get_dummies(df_filtered['age_group_cat'], prefix=None, drop_first=False, dtype=int)
    baseline_age_col = f'{demographic_predictors_prefix}_age_group_middle_aged'
    age_group_dummies_to_add = age_group_dummies_all.copy()
    if baseline_age_col in age_group_dummies_all.columns:
        age_group_dummies_to_add = age_group_dummies_all.drop(columns=[baseline_age_col])
    else:
        age_group_dummies_to_add = pd.get_dummies(df_filtered['age_group_cat'], prefix=None, drop_first=True, dtype=int)
    for col in age_group_dummies_to_add.columns: df_filtered[col] = age_group_dummies_to_add[col]; demographic_dummy_cols_generated.append(col)

gender_col_name = demographic_cols_config.get('gender')
if gender_col_name and gender_col_name in df_filtered.columns:
    df_filtered['gender_processed'] = df_filtered[gender_col_name].str.lower().str.strip().replace({'woman': 'female', 'man': 'male'})
    valid_genders = ['female', 'male']
    df_filtered['gender_for_dummies'] = df_filtered['gender_processed'].apply(lambda x: x if x in valid_genders else np.nan)
    gender_dummies_all = pd.get_dummies(df_filtered['gender_for_dummies'], prefix=f'{demographic_predictors_prefix}_gender', drop_first=False, dtype=int)
    baseline_gender_col = f'{demographic_predictors_prefix}_gender_male'
    gender_dummies_to_add = gender_dummies_all.copy()
    if baseline_gender_col in gender_dummies_all.columns:
        gender_dummies_to_add = gender_dummies_all.drop(columns=[baseline_gender_col])
    elif f'{demographic_predictors_prefix}_gender_female' in gender_dummies_all.columns and len(gender_dummies_all.columns) == 1:
        pass
    elif len(gender_dummies_all.columns) > 1:
        gender_dummies_to_add = pd.get_dummies(df_filtered['gender_for_dummies'], prefix=f'{demographic_predictors_prefix}_gender', drop_first=True, dtype=int)
    for col in gender_dummies_to_add.columns: df_filtered[col] = gender_dummies_to_add[col]; demographic_dummy_cols_generated.append(col)

political_col_name = demographic_cols_config.get('political')
if political_col_name and political_col_name in df_filtered.columns:
    def map_ideology(affiliation):
        """Maps detailed political affiliation strings to 'conservative', 'liberal', or 'neutral'."""
        affiliation_str = str(affiliation).lower()
        if 'republican' in affiliation_str: return 'conservative'
        if 'liberal' in affiliation_str or 'democrat' in affiliation_str: return 'liberal'
        if 'neutral' in affiliation_str or 'moderate' in affiliation_str: return 'neutral'
        return np.nan

    df_filtered['political_group_cat'] = df_filtered[political_col_name].apply(map_ideology)
    political_dummies_all = pd.get_dummies(df_filtered['political_group_cat'], prefix=f'{demographic_predictors_prefix}_poli', drop_first=False, dtype=int)
    baseline_political_col = f'{demographic_predictors_prefix}_poli_neutral'
    political_dummies_to_add = political_dummies_all.copy()
    if baseline_political_col in political_dummies_all.columns:
        political_dummies_to_add = political_dummies_all.drop(columns=[baseline_political_col])
    else:
        political_dummies_to_add = pd.get_dummies(df_filtered['political_group_cat'], prefix=f'{demographic_predictors_prefix}_poli', drop_first=True, dtype=int)
    for col in political_dummies_to_add.columns: df_filtered[col] = political_dummies_to_add[col]; demographic_dummy_cols_generated.append(col)

issue_stance_col_name_csv = demographic_cols_config.get('issue_stance')
if issue_stance_col_name_csv and issue_stance_col_name_csv in df_filtered.columns:
    df_filtered[issue_stance_col_name_csv] = df_filtered[issue_stance_col_name_csv].map(issue_stance_mapping_cleaned_keys).fillna(df_filtered[issue_stance_col_name_csv])
    stance_dummies_all = pd.get_dummies(df_filtered[issue_stance_col_name_csv], prefix=issue_stance_prefix, drop_first=False, dtype=int, dummy_na=False)
    baseline_stance_col_full = f'{issue_stance_prefix}_{baseline_issue_stance.replace(" ", "_")}'
    stance_dummies_to_add = stance_dummies_all.copy()
    if baseline_stance_col_full in stance_dummies_all.columns:
        if len(stance_dummies_all.columns) > 1: stance_dummies_to_add = stance_dummies_all.drop(columns=[baseline_stance_col_full])
    elif len(stance_dummies_all.columns) > 1:
        stance_dummies_to_add = pd.get_dummies(df_filtered[issue_stance_col_name_csv], prefix=issue_stance_prefix, drop_first=True, dtype=int, dummy_na=False)
    for col in stance_dummies_to_add.columns: df_filtered[col] = stance_dummies_to_add[col]; demographic_dummy_cols_generated.append(col)

# --- Prepare Feature DataFrame and Main Effects ---
# This section constructs the main feature DataFrame 'X_build' by creating
# dummy variables for all main effects: topics, emotions, and demographics.
X_build = pd.DataFrame(index=df_filtered.index)
valid_topic_indicator_map = {}
actual_emotion_dummy_names = []

if frequent_topics_orig_names:
    temp_strip_cols = {}
    for tc_col in existing_topic_cols:
        stripped_col_name = f"__stripped_{tc_col}__";
        df_filtered[stripped_col_name] = df_filtered[tc_col].astype(str).str.strip();
        temp_strip_cols[tc_col] = stripped_col_name
    for topic_orig_name in frequent_topics_orig_names:
        indicator_col_name = format_topic_name_for_script(topic_orig_name);
        valid_topic_indicator_map[topic_orig_name] = indicator_col_name
        condition = pd.Series(False, index=df_filtered.index)
        for tc_col in existing_topic_cols: condition = condition | (df_filtered[temp_strip_cols[tc_col]] == topic_orig_name)
        X_build[indicator_col_name] = condition.astype(int)
    for tc_col in existing_topic_cols:
        if temp_strip_cols[tc_col] in df_filtered.columns: df_filtered.drop(columns=[temp_strip_cols[tc_col]], inplace=True)

if emotion_col in df_filtered.columns and df_filtered[emotion_col].nunique() > 0:
    df_filtered[emotion_col] = df_filtered[emotion_col].astype(str)
    emotion_dummies_all = pd.get_dummies(df_filtered[emotion_col], prefix=emotion_dummy_prefix, drop_first=False, dtype=int)
    baseline_col_full_name = f"{emotion_dummy_prefix}_{str(baseline_emotion).lower()}"
    emotion_dummies_to_add = emotion_dummies_all.copy()
    if baseline_col_full_name in emotion_dummies_all.columns:
        if len(emotion_dummies_all.columns) > 1:
            emotion_dummies_to_add = emotion_dummies_all.drop(baseline_col_full_name, axis=1)
    elif len(emotion_dummies_all.columns) > 1:
        emotion_dummies_to_add = pd.get_dummies(df_filtered[emotion_col], prefix=emotion_dummy_prefix, drop_first=True, dtype=int)
    for col in emotion_dummies_to_add.columns: X_build[col] = emotion_dummies_to_add[col]; actual_emotion_dummy_names.append(col)

demographic_main_effect_names = []
for demo_col in demographic_dummy_cols_generated:
    if demo_col in df_filtered.columns: X_build[demo_col] = df_filtered[demo_col]; demographic_main_effect_names.append(demo_col)

# --- Center Predictors and Create Interaction Terms ---
# This block centers the main effect variables that will be used in interactions
# to reduce multicollinearity. It then creates the interaction terms by
# multiplying the centered components.
components_to_potentially_center = set()
if specific_interaction_terms_to_create:
    for term_tuple in specific_interaction_terms_to_create:
        for component_original_name in term_tuple:
            if component_original_name in X_build.columns:
                components_to_potentially_center.add(component_original_name)

final_predictor_names = []
centered_name_map_for_product = {}

for topic_orig_name, topic_indicator_col in valid_topic_indicator_map.items():
    if topic_indicator_col in X_build.columns:
        if topic_indicator_col in components_to_potentially_center:
            centered_col_name = f"{topic_indicator_col}_c";
            X_build[centered_col_name] = X_build[topic_indicator_col] - X_build[topic_indicator_col].mean()
            final_predictor_names.append(centered_col_name);
            centered_name_map_for_product[topic_indicator_col] = centered_col_name
        else:
            final_predictor_names.append(topic_indicator_col);
            centered_name_map_for_product[topic_indicator_col] = topic_indicator_col

for emo_dummy_col in actual_emotion_dummy_names:
    if emo_dummy_col in X_build.columns:
        if emo_dummy_col in components_to_potentially_center:
            centered_col_name = f"{emo_dummy_col}_c";
            X_build[centered_col_name] = X_build[emo_dummy_col] - X_build[emo_dummy_col].mean()
            final_predictor_names.append(centered_col_name);
            centered_name_map_for_product[emo_dummy_col] = centered_col_name
        else:
            final_predictor_names.append(emo_dummy_col);
            centered_name_map_for_product[emo_dummy_col] = emo_dummy_col

for demo_col in demographic_main_effect_names:
    if demo_col in X_build.columns:
        if demo_col in components_to_potentially_center:
            centered_col_name = f"{demo_col}_c";
            X_build[centered_col_name] = X_build[demo_col] - X_build[demo_col].mean()
            final_predictor_names.append(centered_col_name);
            centered_name_map_for_product[demo_col] = centered_col_name
        else:
            final_predictor_names.append(demo_col);
            centered_name_map_for_product[demo_col] = demo_col

all_custom_interaction_term_cols = []
interaction_term_final_frequencies = {}
if specific_interaction_terms_to_create:
    for term_tuple_original_names in specific_interaction_terms_to_create:
        model_ready_component_cols = [];
        valid_term_for_creation = True
        for original_comp_name in term_tuple_original_names:
            if original_comp_name not in centered_name_map_for_product:
                print(f"Warning: Component '{original_comp_name}' for interaction '{term_tuple_original_names}' not found. Interaction cannot be created.")
                valid_term_for_creation = False; break
            model_ready_component_cols.append(centered_name_map_for_product[original_comp_name])
        if not valid_term_for_creation: continue

        interaction_col_name = "_X_".join(model_ready_component_cols)
        product_components_exist = all(comp in X_build.columns for comp in model_ready_component_cols)
        if not product_components_exist:
            print(f"  Warning: Not all model-ready components for interaction '{interaction_col_name}' exist. Skipping term creation.")
            continue

        current_interaction_mask = pd.Series(True, index=X_build.index);
        all_original_components_exist_for_freq = True
        for original_comp_name in term_tuple_original_names:
            if original_comp_name not in X_build.columns: all_original_components_exist_for_freq = False; break
            current_interaction_mask &= (X_build[original_comp_name] == 1)
        combination_frequency = current_interaction_mask.sum() if all_original_components_exist_for_freq else 0
        interaction_term_final_frequencies[interaction_col_name] = combination_frequency

        X_build[interaction_col_name] = X_build[model_ready_component_cols[0]].copy()
        for i in range(1, len(model_ready_component_cols)): X_build[interaction_col_name] *= X_build[model_ready_component_cols[i]]
        final_predictor_names.append(interaction_col_name);
        all_custom_interaction_term_cols.append(interaction_col_name)

# --- Finalize Features and Remove Multicollinearity using VIF ---
y = df_filtered[dependent_var].loc[X_build.index]
final_predictor_names = sorted(list(set(final_predictor_names)))
X_ols_features = X_build[final_predictor_names].copy()

X_ols_features.dropna(inplace=True)
y = y.loc[X_ols_features.index]

if not X_ols_features.empty:
    print(f"\n--- Starting Iterative VIF-based Feature Selection (Threshold > 5) ---")
    active_predictors = X_ols_features.columns.tolist()
    max_iterations = len(active_predictors) + 5
    iteration = 0
    while len(active_predictors) > 1 and iteration < max_iterations:
        iteration += 1
        if not active_predictors or len(active_predictors) <= 1: break
        temp_X_vif = X_ols_features[active_predictors].astype(float)
        X_vif_calc = sm.add_constant(temp_X_vif, has_constant='add')

        try:
            vif_values = pd.Series([variance_inflation_factor(X_vif_calc.values, i + 1) for i in range(len(active_predictors))], index=active_predictors, name='VIF')
        except Exception as e:
            print(f"Error during VIF calculation on iteration {iteration}: {e}. Predictors: {active_predictors}"); break

        max_vif = vif_values.max()
        if max_vif > 5:
            feature_to_remove = vif_values.idxmax()
            print(f"  Iter {iteration}: Removing '{feature_to_remove}' with VIF: {max_vif:.2f}")
            active_predictors.remove(feature_to_remove)
            if not active_predictors: break
        else:
            print(f"  Iter {iteration}: No remaining feature with VIF > 5. Max VIF: {max_vif:.2f}. Stopping."); break
    if iteration >= max_iterations: print(" VIF removal stopped: Max iterations reached.")
    elif len(active_predictors) <= 1 and iteration > 0 and X_ols_features.shape[1] > 1: print(" VIF check stopped: 1 or fewer predictors remaining.")

    X_ols_features = X_ols_features[active_predictors].copy()
    y = y.loc[X_ols_features.index]
    print(f"Number of predictors after VIF selection: {len(X_ols_features.columns)}")
else:
    print("X_ols_features is empty before VIF check, skipping VIF removal.")

X_for_vif_report = sm.add_constant(X_ols_features.copy(), has_constant='add') if not X_ols_features.empty else pd.DataFrame()
X = sm.add_constant(X_ols_features.copy(), has_constant='add') if not X_ols_features.empty else sm.add_constant(pd.DataFrame(index=y.index))

# --- Fit OLS Model and Perform F-Test ---
# This block fits the final OLS model with the selected predictors. It prints the model summary,
# a final VIF report, and conducts an omnibus F-test to assess the
# overall significance of the block of interaction terms.
results = None
if not X_ols_features.empty or ('const' in X.columns and X.shape[1] == 1):
    try:
        model = sm.OLS(y, X)
        results = model.fit()
        print(f"--- OLS Regression Results (Full Model) ---"); print(results.summary())
        print(f"\n--- Variance Inflation Factor (VIF) for Full Model Predictors ---")
        if not X_ols_features.empty:
            if X_ols_features.shape[1] > 1:
                vif_data = pd.DataFrame(); vif_data["feature"] = X_ols_features.columns
                vif_data["VIF"] = [variance_inflation_factor(X_for_vif_report.values, i + 1) for i in range(len(X_ols_features.columns))]
                print(vif_data.sort_values("VIF", ascending=False))
            elif X_ols_features.shape[1] == 1:
                print(f"Only one predictor ('{X_ols_features.columns[0]}') remains. VIF is not applicable.")
        else:
            print("X_ols_features is empty, VIF report skipped.")
    except Exception as e:
        print(f"An error occurred during model fitting: {e}"); results = None

    if results is not None and not X_ols_features.empty:
        interaction_terms_in_final_model = [p for p in X_ols_features.columns if "_X_" in p]
        if interaction_terms_in_final_model:
            print(f"\n--- Omnibus F-test for the block of {len(interaction_terms_in_final_model)} Interaction Term(s) ---")
            main_effects_in_final_model = [p for p in X_ols_features.columns if "_X_" not in p]
            if main_effects_in_final_model:
                X_reduced_features = X_ols_features[main_effects_in_final_model].copy()
                X_reduced = sm.add_constant(X_reduced_features, has_constant='add')
                try:
                    model_reduced = sm.OLS(y, X_reduced)
                    results_reduced = model_reduced.fit()
                    anova_results_interactions = anova_lm(results_reduced, results)
                    print(anova_results_interactions)
                except Exception as e:
                    print(f"Error during omnibus F-test for interactions: {e}")
            else:
                print("Skipping omnibus F-test as no main effects are available for the reduced model.")
        elif X_ols_features.columns.tolist():
            print("\nNo interaction terms were included in the final model. Skipping omnibus test.")
else:
    print("Skipping model fitting as no predictors remain or data is insufficient."); results = None

# === Part 2: Plotting ===

# --- Calculate Predictor Frequencies for Plotting ---
# For each predictor in the final model, this block calculates its frequency (N)
# in the dataset. This is used for annotating the forest plots.
predictor_frequencies = {}
if 'topic_counts_final' in locals() and topic_counts_final is not None:
    for topic_orig_name, indicator_col_name in valid_topic_indicator_map.items():
        count = topic_counts_final.get(topic_orig_name, 0);
        if indicator_col_name in X_ols_features.columns: predictor_frequencies[indicator_col_name] = count
        centered_name = f"{indicator_col_name}_c"
        if centered_name in X_ols_features.columns: predictor_frequencies[centered_name] = count

if emotion_counts_in_filtered is not None:
    for dummy_name in actual_emotion_dummy_names:
        if dummy_name.startswith(emotion_dummy_prefix + "_"):
            orig_emotion_value = dummy_name.replace(emotion_dummy_prefix + "_", "", 1)
            count = emotion_counts_in_filtered.get(orig_emotion_value, 0);
            if dummy_name in X_ols_features.columns: predictor_frequencies[dummy_name] = count
            centered_name = f"{dummy_name}_c"
            if centered_name in X_ols_features.columns: predictor_frequencies[centered_name] = count

for demo_col_name in demographic_main_effect_names:
    original_in_model = demo_col_name in X_ols_features.columns
    centered_name = f"{demo_col_name}_c"
    centered_in_model = centered_name in X_ols_features.columns
    model_col_to_use_for_freq_key = None
    if original_in_model: model_col_to_use_for_freq_key = demo_col_name
    if centered_in_model: model_col_to_use_for_freq_key = centered_name
    if model_col_to_use_for_freq_key:
        if demo_col_name in X_build.columns and not X_ols_features.empty:
            count = X_build[demo_col_name].loc[X_ols_features.index].sum()
            predictor_frequencies[model_col_to_use_for_freq_key] = int(count)
        else:
            predictor_frequencies[model_col_to_use_for_freq_key] = 0

for int_term_model_name, count in interaction_term_final_frequencies.items():
    if int_term_model_name in X_ols_features.columns:
        predictor_frequencies[int_term_model_name] = count

final_model_frequencies = {k: v for k, v in predictor_frequencies.items() if k in X_ols_features.columns or k == 'const'}
predictor_frequencies_series = pd.Series(final_model_frequencies).fillna(0).astype(int)

# --- Create Display Name Mappings for Plots ---
# This block creates a mapping from the script's internal variable names
# (e.g., 'topic_Economy_Jobs_c') to human-readable labels for the plots (e.g., 'Economy & Jobs').
display_name_map = {'const': 'Intercept'}
for topic_orig_name, script_topic_name in valid_topic_indicator_map.items():
    display_name_map[script_topic_name] = topic_orig_name
    display_name_map[f"{script_topic_name}_c"] = topic_orig_name
for emo_dummy_name in actual_emotion_dummy_names:
    if emo_dummy_name.startswith(emotion_dummy_prefix + "_"):
        orig_emotion_value = emo_dummy_name.replace(emotion_dummy_prefix + "_", "", 1);
        display_emotion = orig_emotion_value.capitalize()
        display_name_map[emo_dummy_name] = display_emotion
        display_name_map[f"{emo_dummy_name}_c"] = display_emotion
for demo_col_original_name in demographic_main_effect_names:
    model_col_name = demo_col_original_name
    if f"{demo_col_original_name}_c" in X_ols_features.columns:
        model_col_name = f"{demo_col_original_name}_c"
    elif demo_col_original_name not in X_ols_features.columns and model_col_name != 'const':
        continue
    name_to_display = demo_col_original_name
    if demo_col_original_name.startswith(demographic_predictors_prefix + "_"):
        temp_name = demo_col_original_name.replace(f"{demographic_predictors_prefix}_", "")
        if temp_name.startswith("age_group_"): name_to_display = f"Age: {temp_name.replace('age_group_', '').replace('_', '-').capitalize()}"
        elif temp_name.startswith("gender_"): name_to_display = f"Gender: {temp_name.replace('gender_', '').capitalize()}"
        elif temp_name.startswith("poli_"): name_to_display = f"Politics: {temp_name.replace('poli_', '').capitalize()}"
    elif demo_col_original_name.startswith(issue_stance_prefix + "_"):
        value_part = demo_col_original_name.replace(issue_stance_prefix + '_', '')
        name_to_display = f"Stance: {value_part.replace('_', ' ').capitalize()}"
    display_name_map[model_col_name] = name_to_display
for interaction_term_model_name in X_ols_features.columns:
    if "_X_" not in interaction_term_model_name: continue
    parts_in_model_name = interaction_term_model_name.split('_X_')
    display_parts = [display_name_map.get(model_part_name, model_part_name) for model_part_name in parts_in_model_name]
    display_name_map[interaction_term_model_name] = " * ".join(display_parts)


if results is not None:
    def create_forest_plot(results, specific_vars_list=None, title="Forest Plot", frequencies=None, display_name_map=None, significance_threshold=0.05, figsize=(10, 8), sort_by_coef=True, only_significant=False):
        """Generates and displays a forest plot for OLS regression results, showing coefficients and confidence intervals."""
        params = results.params.copy();
        conf_int_df = results.conf_int().copy();
        p_values = results.pvalues.copy()
        if 0 in conf_int_df.columns and 1 in conf_int_df.columns:
            conf_int_df = conf_int_df.rename(columns={0: 'ci_lower', 1: 'ci_upper'})
        elif len(conf_int_df.columns) == 2:
            conf_int_df.columns = ['ci_lower', 'ci_upper']
        else:
            print(f"Warning: Unexpected CI columns for plot '{title}'."); return
        plot_data = pd.DataFrame({'coef': params, 'pvalue': p_values}).join(conf_int_df)

        vars_in_model = results.params.index.tolist()
        valid_specific_vars = [var for var in specific_vars_list if var in vars_in_model] if specific_vars_list is not None else [var for var in vars_in_model if var != 'const']
        if not valid_specific_vars: print(f"Warning: No valid variables for plot '{title}' found."); return
        plot_data = plot_data.loc[valid_specific_vars]

        if 'const' in plot_data.index: plot_data = plot_data.drop('const')
        if plot_data.empty: print(f"No variables to plot for '{title}'."); return
        if only_significant: plot_data = plot_data[plot_data['pvalue'] < significance_threshold]
        if plot_data.empty: print(f"No {'significant ' if only_significant else ''}variables to plot for '{title}'."); return

        plot_data['freq'] = plot_data.index.map(frequencies if isinstance(frequencies, pd.Series) else pd.Series(frequencies)).fillna(0).astype(int) if frequencies is not None else 0
        plot_data_sorted = plot_data.sort_values('coef', ascending=True) if sort_by_coef else plot_data.reindex(plot_data['coef'].abs().sort_values(ascending=False).index)
        plot_display_labels_sorted = [display_name_map.get(idx_name, idx_name) for idx_name in plot_data_sorted.index]
        plot_data_sorted['ci_width'] = plot_data_sorted['ci_upper'] - plot_data_sorted['ci_lower']

        fig, ax = plt.subplots(figsize=figsize);
        y_pos = np.arange(len(plot_data_sorted))
        ax.errorbar(plot_data_sorted['coef'], y_pos, xerr=plot_data_sorted['ci_width'] / 2, fmt='o', color='black', capsize=5, linestyle='')
        ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)
        x_min_plot, x_max_plot = ax.get_xlim();
        x_range_plot = x_max_plot - x_min_plot if x_max_plot > x_min_plot else 1
        horizontal_offset = x_range_plot * 0.02

        for i in range(len(plot_data_sorted)):
            row_data = plot_data_sorted.iloc[i];
            freq = row_data['freq']
            if freq > 0:
                text_x_coord = row_data['ci_upper'] + horizontal_offset; ha_align = 'left'
                text_render_width_approx = len(f"N={freq}") * 0.007 * x_range_plot
                if text_x_coord + text_render_width_approx > x_max_plot:
                    text_x_coord = row_data['ci_lower'] - horizontal_offset; ha_align = 'right'
                if text_x_coord - text_render_width_approx < x_min_plot and ha_align == 'right':
                    text_x_coord = row_data['coef'] + horizontal_offset
                    ha_align = 'left' if row_data['coef'] < (x_min_plot + x_max_plot) / 2 else 'right'
                ax.text(text_x_coord, y_pos[i], f"N={freq}", va='center', ha=ha_align, fontsize=7, color='dimgray')

        ax.set_yticks(y_pos); ax.set_yticklabels(plot_display_labels_sorted); ax.invert_yaxis()
        ax.set_xlabel(f'Coefficient (Effect on {dependent_var})'); ax.set_ylabel('Predictor Variable')
        ax.set_title(f"{title}\n{'(P < ' + str(significance_threshold) + ')' if only_significant else '(All Shown)'}")
        plt.grid(axis='x', linestyle=':', linewidth=0.5); plt.tight_layout(); plt.show()


    # --- Generate and Display Forest Plots ---
    # This final section prepares lists of different variable types (e.g., topics, interactions)
    # and calls the plotting function to generate a separate forest plot for each category.
    topic_main_effects_plot = [p for p in X_ols_features.columns if p.startswith('topic_') and "_X_" not in p]
    emotion_main_effects_plot = [p for p in X_ols_features.columns if p.startswith(emotion_dummy_prefix + "_") and "_X_" not in p]
    demographic_effects_plot_no_stance = [p for p in X_ols_features.columns if p.startswith(demographic_predictors_prefix) and "_X_" not in p]
    stance_main_effects_plot = [p for p in X_ols_features.columns if p.startswith(issue_stance_prefix + "_") and "_X_" not in p]
    interaction_effects_plot = [p for p in X_ols_features.columns if "_X_" in p]

    if topic_main_effects_plot: create_forest_plot(results, specific_vars_list=topic_main_effects_plot, title='Forest Plot: Significant Main Effects of Topics', frequencies=predictor_frequencies_series, display_name_map=display_name_map, figsize=(10, max(6, len(topic_main_effects_plot) * 0.4)), only_significant=True)
    if emotion_main_effects_plot: create_forest_plot(results, specific_vars_list=emotion_main_effects_plot, title=f'Forest Plot: Main Effects of Emotions (vs. baseline {baseline_emotion})', frequencies=predictor_frequencies_series, display_name_map=display_name_map, figsize=(10, max(6, len(emotion_main_effects_plot) * 0.45)), only_significant=False)
    if interaction_effects_plot: create_forest_plot(results, specific_vars_list=interaction_effects_plot, title=f'Forest Plot: Custom Interaction Effects (Stance * Topic)', frequencies=predictor_frequencies_series, display_name_map=display_name_map, figsize=(10, max(8, len(interaction_effects_plot) * 0.4)), only_significant=False)
    if demographic_effects_plot_no_stance: create_forest_plot(results, specific_vars_list=demographic_effects_plot_no_stance, title='Forest Plot: Main Effects of Demographics (excl. Stance)', frequencies=predictor_frequencies_series, display_name_map=display_name_map, figsize=(10, max(6, len(demographic_effects_plot_no_stance) * 0.45)), only_significant=False)
else:
    print("\n--- Skipping Plot Generation: OLS model fitting did not complete successfully. ---")