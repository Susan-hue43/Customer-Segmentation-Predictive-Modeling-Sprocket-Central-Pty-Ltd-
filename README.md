# Customer Segmentation & Predictive Modeling – Sprocket Central Pty Ltd  

## Introduction
As part of the **KPMG Data Analytics Virtual Internship**, I've worked on a series of projects designed to mirror real-world business challenges. The journey began with [**Project Sprocket Central Data Cleaning**](https://github.com/Susan-hue43/-Sprocket-Data-Cleaning-Project.git), where the focus was on ensuring data quality and preparing it for deeper analysis. That groundwork laid the foundation for this project, which shifts gears toward **predictive modeling and customer segmentation**.

For businesses like **Sprocket Central Pty Ltd**, customer segmentation is critical—it allows marketing efforts and resources to be directed toward the most valuable prospects, increasing efficiency and profitability. By leveraging historical purchase and demographic data, the objective here is to identify **high-value customers** from a new pool of 1,000 prospects.

The datasets had already undergone cleaning, except for a few `dob` null values deliberately retained for observation. For the modeling stage, however, these nulls were removed to maintain consistency and reliability in the predictions.

---

## Table of Contents  
1. [Introduction](#introduction)  
2. [Objective](objective)  
3. [Skills Demonstrated](skills-demonstrated)  
4. [Tools Used](tools-used)  
5. [Data Overview](data-overview)  
6. [Data Cleaning & Feature Engineering](data-cleaning--feature-engineering)  
7. [Exploratory Data Analysis (EDA)](exploratory-data-analysis-eda)  
8. [Modeling Approach](modeling-approach)  
9. [Model Evaluation](model-evaluation)  
10. [Key Findings](key-findings)  
11. [Recommendations](recommendations)  
12. [Conclusion](conclusion)  
 

--- 

## Objective  

The main objective of this project is **Identifying High-Value Customers from 1,000 New Prospects** for Sprocket Central Pty Ltd.  

Using predictive modeling techniques, the goal is to segment potential customers into value tiers **(High, Medium, Low)** based on their historical purchasing behavior. This helps the business prioritize marketing efforts, allocate resources efficiently, and maximize return on investment.  


---

## Skills Demonstrated  
- Data Cleaning & Preprocessing.  
- Feature Engineering.  
- Exploratory Data Analysis (EDA).  
- Predictive Modeling (Classification with XGBoost & Random Forest.  
- Model Evaluation (Accuracy, F1-Score, Quantile Analysis).  
- Customer Segmentation using Purchase Quantiles.  
- Data Visualization & Business Reporting.  

---

## Tools Used  
- **Excel** → for initial raw files and inspection.  
- **VS Code** → development environment.  
- **Python** → data analysis, feature engineering, modeling, and evaluation.  

---

## Data Cleaning & Feature Engineering  
- Data cleaning was performed in the [**Project Sprocket Central Data Cleaning**](https://github.com/Susan-hue43/-Sprocket-Data-Cleaning-Project.git).  
- In this stage:  
  - Dropped `dob` null values retained earlier.
    ```python
    # remove null values
    customers = customers.dropna()
    ``` 
  - **Feature Engineering:**
      1. Engineered **Sales Segment** using purchase quantile cutoffs (0.33, 0.66, max).
      
    ```python
    # create target variable

    bins = [0,
        customers['past_3_years_bike_related_purchases'].quantile(0.33),
        customers['past_3_years_bike_related_purchases'].quantile(0.66),
        customers['past_3_years_bike_related_purchases'].max()]

    labels = ['Low Value', 'Medium Value', 'High Value']
    
    customers['sales_segment'] = pd.cut(customers['past_3_years_bike_related_purchases'], bins=bins, labels=labels, include_lowest=True)
    ```
      2. **Tenure Squared (`tenure_squared`)**- Capture potential non-linear relationships between tenure and purchase behavior.
    
      3. **Log-Transformed Property Valuation (`log_property_valuation`)**- Normalize skewed property valuation values and reduce the effect of outliers.  

      4. **Purchases per Year (`purchase_per_year`)**- Standardize purchase activity relative to customer tenure, ensuring fairness when comparing newer vs. older customers.  

      5. **High-Value Car Owner (`high_value_car_owner`)**- Flag customers with both car ownership and high net worth, as they may represent premium target segments. 

      
      ```python
      # Tenure squared
      customers_fe = customers.copy()
      customers_fe['tenure_squared'] = customers_fe['tenure'] ** 2

      # Log-transformed property valuation
      customers_fe['log_property_valuation'] = np.log1p(customers_fe['property_valuation'])

      # Purchases per year
      customers_fe['purchase_per_year'] = (
          customers_fe['past_3_years_bike_related_purchases'] /
          customers_fe['tenure'].replace(0, np.nan)
      ).fillna(0)

      # High value car owner
      customers_fe['high_value_car_owner'] = np.where(
          (customers_fe['owns_car'] == 'Yes') & (customers_fe['wealth_segment'] == 'High Net Worth'),
          1, 0
      )

      ```
These engineered features were included to improve predictive accuracy and better reflect customer behaviors relevant to purchase patterns.

  - Encoded categorical features and standardized data for modeling.  
     ```python
      # Identify categorical and numeric columns
      cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
      num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

      # Preprocessor
      preprocessor = ColumnTransformer(
          transformers=[
              ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
              ('num', StandardScaler(), num_features)
          ],
          verbose_feature_names_out=False
      )
     ```  

---

## Exploratory Data Analysis (EDA)  
- Distribution of past 3-year purchases by customer.

  <img width="540" height="468" alt="image" src="https://github.com/user-attachments/assets/07ebdf21-de08-4b89-bf75-5295f7368fe1" />

   The past 3-year purchase distribution shows a **balanced customer base** across all segments:

    * **High Value:** 948
    * **Medium Value:** 926
    * **Low Value:** 902

   No single group dominates, indicating that customers contribute relatively evenly to overall sales, with only slight variation between segments.
---
- Gender breakdown of purchases.

<img width="549" height="468" alt="image" src="https://github.com/user-attachments/assets/2b3efad5-8241-4c2b-9cdc-ada9cb881afc" />
  
  The gender distribution of purchases shows a fairly even split, with **females (1,444 purchases)** slightly outnumbering **males (1,331 purchases)**. Only **1   purchase** was recorded under the “unsure” category, indicating minimal ambiguity in gender reporting.

---
- State-level customer concentration.

<img width="549" height="468" alt="image" src="https://github.com/user-attachments/assets/c79272f2-ef2b-4852-9618-2fead5461a3b" />


  The majority of purchases came from **New South Wales (1,484)**, making it the leading state in bike-related purchases over the past three years. **Victoria (708 purchases)** follows as the second-largest contributor, while **Queensland (584 purchases)** recorded the lowest purchase count among the three states.

--- 

## Modeling Approach  

### 1. Model Used  
- **XGBoost Classifier** (multi-class classification) was chosen for its efficiency, handling of non-linear relationships, and robustness with tabular data.  
- Target variable: **Sales Segment**, defined using quartiles of past 3 years’ purchases.  

### 2. Model Development  
- **Data Split:** Training and testing sets were prepared to evaluate generalization.
  
- **Feature Set:** Included demographic, behavioral, and engineered features (e.g., tenure_squared, log_property_valuation, purchase_per_year, high_value_car_owner).  

- **Hyperparameter Tuning:** Applied Grid search to optimize model depth, learning rate, and estimators.

- **Model Training:** The engineered dataset was fitted into XGBoost for multi-class classification.

  - **Prediction on Validation Set:** The trained model was tested against the held-out validation set.  

- **Evaluation:** Performance was assessed using:  
  - **Confusion Matrix** – to visualize misclassifications across quartile segments.  

  - **Classification Report** – providing precision, recall, and F1-score for each class.

```python
# Drop unneeded columns
drop_cols = ['customer_id', 'first_name', 'last_name', 'DOB', 'job_title', 'deceased_indicator',
             'address', 'postcode', 'country', 'past_3_years_bike_related_purchases', 'owns_car', 'property_valuation']

target = 'sales_segment'

# Features and target
X = customers_fe.drop(columns=drop_cols + [target])
y = customers_fe[target]

# Encode target for multi-class
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Build XGBoost pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        random_state=42,
        objective='multi:softmax',  # for multi-class
        eval_metric='mlogloss'
    ))
])

# Hyperparameter grid 
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__subsample': [0.8, 1],
    'classifier__colsample_bytree': [0.8, 1]
}

# GridSearchCV with F1 scoring 
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Evaluate on validation set
xgb_pred = grid_search.predict(X_val)
print("\nValidation Performance:\n")
print(classification_report(y_val, xgb_pred, target_names=le_target.classes_))

best_xgb = grid_search.best_estimator_

# Confusion matrix
cm = confusion_matrix(y_val, xgb_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_target.classes_)
disp.plot(cmap='Blues')
plt.title('XGBoost Multi-class Confusion Matrix')
plt.show()

```  
---

### Results and Interpretations
**Best Hyperparameters:**

* `n_estimators`: 100, `max_depth`: 5, `learning_rate`: 0.1, `colsample_bytree`: 1, `subsample`: 1

---

**Classification Report**

<img width="336" height="152" alt="Screenshot 2025-08-20 002207" src="https://github.com/user-attachments/assets/859b68b2-b124-4982-a5ca-65a24139a572" />

* **High Value: F1** = **0.99 (recall 0.99)** → the model almost perfectly identifies high-value customers.
* **Low Value: F1** = **0.99 (recall 1.00)** → predictions for low-value customers are extremely accurate.
* **Medium Value: F1** = **0.98 (recall 0.97)** → the model is highly reliable in classifying medium-value customers.
* **Overall Accuracy: 99%** → the model gets 99 out of 100 customers correctly classified.

---

**Confusion Matrix**

<img width="596" height="453" alt="image" src="https://github.com/user-attachments/assets/fb543c03-0789-449d-b914-ec2560dccfaa" />

* The model makes very few mistakes across all categories.
* Almost all **High Value** customers are correctly classified, with just 2 misclassified.
* **Low Value** customers were classified perfectly, with no errors.
* A small number of **Medium Value** customers (5 total) were misclassified, but the vast majority were correct.

---


## Predictions on New Customers
```python
# Predict
predictions = best_xgb.predict(new_customers_clean)
probs = best_xgb.predict_proba(new_customers_clean)


# Mapping from numeric class to segment labels
segment_map = {0: "High Value", 1: "Low Value", 2: "Medium Value"}

new_customers_clean["Sales_Segment"] = [segment_map[p] for p in predictions]


# Attach the highest probability score for reference
new_customers_clean["Max_Prob"] = probs.max(axis=1)

# Concatenate with original df
final_customers = pd.concat(
    [new_customers.reset_index(drop=True), 
    new_customers_clean[['Sales_Segment', 'Max_Prob']]], axis= 1
)

# Save to Excel
output_path = "NewCustomerList_with_salessegments.xlsx"
final_customers.to_excel(output_path, index=False)

print("✅ Predictions with Sales_Segment saved to", output_path)
```

## Model Evaluation 

| Quantile         | Training Data | New Customers |
|------------------|---------------|---------------|
| 0.33 Quantile    | 33.0          | 36.0          |
| 0.66 Quantile    | 66.0          | 64.0          |
| Max              | 99.0          | 99.0          |


Model evaluation was conducted using a **Quantile Cutoff Comparison**, where predicted customer segments were compared with actual purchase quartiles. This ensured that the model’s classifications aligned with real purchase behavior.

**Key Insights:**

* The model successfully matched predicted **High Value** customers with the top spending quartile.
* **Medium Value** predictions closely aligned with middle purchase quartiles, showing consistency.
* **Low Value** predictions fell into the lowest quartile as expected.

**Overall Result:**
The quantile comparison confirmed that the model’s segmentation strongly reflects real-world spending behavior, giving confidence in the predictions made on new customers.

**Interpretation:** The predicted new customer purchase segments closely match the training distribution, suggesting the model generalizes well.  

---

## Key Findings

The model segmented the **1,000 new customers** into three value groups:

* **Medium Value: 366** customers.
* **High Value: 328** customers.
* **Low Value: 306** customers.

<img width="540" height="468" alt="image" src="https://github.com/user-attachments/assets/1e4d390d-5eba-4239-9a34-c4af59eea3f7" />


These results show that the largest portion of customers falls into the **Medium Value** segment, followed closely by **High Value** and **Low Value** groups, with all three segments relatively balanced in distribution.

---
## Conclusion  
The predictive modeling successfully segmented the new customer list in line with historical patterns. This provides Sprocket Central Pty Ltd with a reliable **customer targeting framework** to maximize ROI from marketing and sales campaigns.  

---

## Recommendations  
- **Prioritize High-Value Customers**: Focus marketing and premium offers here.  
- **Engage Medium-Value Customers**: Target them with promotions to move them into the high-value segment.  
- **Optimize Resources for Low-Value Customers**: Maintain engagement but with minimal spend.  

---
