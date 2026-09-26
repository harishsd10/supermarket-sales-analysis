# Supermarket Sales Analysis & Supervised Machine Learning Pipeline
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    csv_path = os.path.join(base_dir, 'supermarket_sales.csv')
    print('=' * 70)
    print(' SUPERMARKET SALES ANALYSIS & MACHINE LEARNING PIPELINE')
    print('=' * 70)
    print(f'[1/7] Loading dataset from: {csv_path}')
    df = pd.read_csv(csv_path)
    print(f'      Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns')
    
    # Preprocessing
    print('\n[2/7] Preprocessing & Feature Engineering...')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month_name()
    df['Day'] = df['Date'].dt.day_name()
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
    
    def get_time_of_day(hour):
        if 10 <= hour < 14:
            return 'Morning / Noon'
        elif 14 <= hour < 18:
            return 'Afternoon'
        else:
            return 'Evening'
            
    df['TimeOfDay'] = df['Hour'].apply(get_time_of_day)
    df['Rating_Category'] = (df['Rating'] >= 7.0).astype(int)
    df['Rating_Label'] = df['Rating_Category'].map({1: 'High (>= 7.0)', 0: 'Low (< 7.0)'})
    
    processed_csv = os.path.join(base_dir, 'processed_supermarket_data.csv')
    df.to_csv(processed_csv, index=False)
    print(f'      Saved processed CSV to: {processed_csv}')
    
    # KPIs
    print('\n[3/7] Core Metrics:')
    print(f'      * Total Sales         : ')
    print(f'      * Total Gross Income  : ')
    print(f'      * Total Transactions  : {len(df):,}')
    print(f'      * Average Order Value : ')
    print(f'      * Average Rating      : {df["Rating"].mean():.2f}/10')
    print(f'      * High Rating Ratio   : {(df["Rating_Category"].mean()*100):.1f}%')
    
    # Visualizations
    print('\n[4/7] Generating EDA Visualizations...')
    
    # 1. Product Lines
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    prod_rev = df.groupby('Product line')['Total'].sum().sort_values(ascending=False)
    prod_qty = df.groupby('Product line')['Quantity'].sum().sort_values(ascending=False)
    
    ax[0].barh(prod_rev.index[::-1], prod_rev.values[::-1], color=['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'][::-1], edgecolor='black', alpha=0.85)
    ax[0].set_title('Total Revenue by Product Line', fontweight='bold')
    ax[0].set_xlabel('Revenue ($)')
    for i, v in enumerate(prod_rev.values[::-1]):
        ax[0].text(v + 1000, i, f'', va='center', fontweight='bold', fontsize=10)
        
    ax[1].barh(prod_qty.index[::-1], prod_qty.values[::-1], color='#6366F1', edgecolor='black', alpha=0.85)
    ax[1].set_title('Total Quantity Sold by Product Line', fontweight='bold')
    ax[1].set_xlabel('Units Sold')
    for i, v in enumerate(prod_qty.values[::-1]):
        ax[1].text(v + 15, i, f'{v:,} units', va='center', fontweight='bold', fontsize=10)
        
    plt.tight_layout()
    fig.savefig(os.path.join(img_dir, 'eda_product_lines.png'), dpi=300)
    plt.close()
    
    # 2. Branch & City
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    branch_perf = df.groupby(['Branch', 'City']).agg({'Total': 'sum', 'Rating': 'mean'}).reset_index()
    branch_perf['Branch_City'] = branch_perf['Branch'] + ' (' + branch_perf['City'] + ')'
    
    bars = ax[0].bar(branch_perf['Branch_City'], branch_perf['Total'], color=['#3B82F6', '#10B981', '#F59E0B'], edgecolor='black', width=0.55)
    ax[0].set_title('Total Revenue by Branch & City', fontweight='bold')
    ax[0].set_ylabel('Revenue ($)')
    for bar in bars:
        height = bar.get_height()
        ax[0].text(bar.get_x() + bar.get_width()/2., height + 2000, f'', ha='center', fontweight='bold')
        
    bars2 = ax[1].bar(branch_perf['Branch_City'], branch_perf['Rating'], color=['#60A5FA', '#34D399', '#FBBF24'], edgecolor='black', width=0.55)
    ax[1].set_title('Average Customer Rating by Branch', fontweight='bold')
    ax[1].set_ylabel('Rating (out of 10)')
    ax[1].set_ylim(0, 10)
    for bar in bars2:
        height = bar.get_height()
        ax[1].text(bar.get_x() + bar.get_width()/2., height + 0.2, f'{height:.2f}', ha='center', fontweight='bold')
        
    plt.tight_layout()
    fig.savefig(os.path.join(img_dir, 'eda_branch_city.png'), dpi=300)
    plt.close()
    
    # 3. Customer Demographics
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    cust_gender = df.groupby(['Customer type', 'Gender'])['Total'].sum().unstack()
    cust_gender.plot(kind='bar', ax=ax[0], color=['#EC4899', '#3B82F6'], edgecolor='black', width=0.6)
    ax[0].set_title('Revenue by Customer Type & Gender', fontweight='bold')
    ax[0].set_ylabel('Total Revenue ($)')
    ax[0].tick_params(axis='x', rotation=0)
    
    aov_segment = df.groupby(['Customer type', 'Gender'])['Total'].mean().unstack()
    aov_segment.plot(kind='bar', ax=ax[1], color=['#F472B6', '#60A5FA'], edgecolor='black', width=0.6)
    ax[1].set_title('Average Order Value (AOV) by Segment', fontweight='bold')
    ax[1].set_ylabel('AOV ($)')
    ax[1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    fig.savefig(os.path.join(img_dir, 'eda_customer_behavior.png'), dpi=300)
    plt.close()
    
    # 4. Payment Methods
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    pay_counts = df['Payment'].value_counts()
    ax[0].pie(pay_counts, labels=pay_counts.index, autopct='%1.1f%%', colors=['#10B981', '#3B82F6', '#F59E0B'], explode=(0.03, 0.03, 0.03), startangle=140, textprops={'fontweight': 'bold'})
    ax[0].set_title('Payment Method Distribution (% of Orders)', fontweight='bold')
    
    pay_summary = df.groupby('Payment')['Total'].agg(['sum', 'mean'])
    bars_pay = ax[1].bar(pay_summary.index, pay_summary['sum'], color=['#10B981', '#3B82F6', '#F59E0B'], edgecolor='black', width=0.5)
    ax[1].set_title('Total Revenue by Payment Method', fontweight='bold')
    ax[1].set_ylabel('Total Sales ($)')
    for bar in bars_pay:
        height = bar.get_height()
        ax[1].text(bar.get_x() + bar.get_width()/2., height + 2000, f'', ha='center', fontweight='bold')
        
    plt.tight_layout()
    fig.savefig(os.path.join(img_dir, 'eda_payment_methods.png'), dpi=300)
    plt.close()
    
    # 5. Rating & Hourly
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.histplot(df['Rating'], kde=True, bins=15, color='#8B5CF6', ax=ax[0])
    ax[0].axvline(7.0, color='red', linestyle='--', linewidth=2, label='Threshold 7.0')
    ax[0].set_title('Customer Rating Distribution', fontweight='bold')
    ax[0].set_xlabel('Rating (4.0 - 10.0)')
    ax[0].legend()
    
    hourly_sales = df.groupby('Hour')['Total'].sum()
    ax[1].plot(hourly_sales.index, hourly_sales.values, marker='o', color='#EF4444', linewidth=2.5, markersize=8)
    ax[1].set_title('Hourly Sales Trend (Peak Shopping Hours)', fontweight='bold')
    ax[1].set_xlabel('Hour of Day (24h)')
    ax[1].set_ylabel('Total Revenue ($)')
    ax[1].set_xticks(hourly_sales.index)
    ax[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(os.path.join(img_dir, 'eda_rating_distribution.png'), dpi=300)
    plt.close()
    
    # ML Preparation
    print('\n[5/7] ML Feature Encoding & Train-Test Split...')
    feature_cols = [
        'Branch', 'City', 'Customer type', 'Gender', 'Product line',
        'Unit price', 'Quantity', 'Tax 5%', 'Total', 'Payment',
        'cogs', 'gross income', 'Hour', 'DayOfWeek'
    ]
    
    df_model = df.copy()
    cat_cols = ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment']
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        
    X = df_model[feature_cols]
    y = df_model['Rating_Category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Training Models
    print('\n[6/7] Training Models...')
    dt_model = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_split=10, min_samples_leaf=5, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    y_prob_dt = dt_model.predict_proba(X_test)[:, 1]
    
    lr_model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    acc_dt = accuracy_score(y_test, y_pred_dt)
    prec_dt = precision_score(y_test, y_pred_dt)
    rec_dt = recall_score(y_test, y_pred_dt)
    f1_dt = f1_score(y_test, y_pred_dt)
    auc_dt = roc_auc_score(y_test, y_prob_dt)
    
    acc_lr = accuracy_score(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr)
    rec_lr = recall_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    
    print('\n' + '=' * 70)
    print(' MODEL PERFORMANCE BENCHMARK MATRIX')
    print('=' * 70)
    print(f"{'Metric':<20} | {'Decision Tree Classifier':<25} | {'Logistic Regression':<20}")
    print('-' * 70)
    print(f"{'Accuracy':<20} | {acc_dt:<25.4f} | {acc_lr:<20.4f}")
    print(f"{'Precision':<20} | {prec_dt:<25.4f} | {prec_lr:<20.4f}")
    print(f"{'Recall':<20} | {rec_dt:<25.4f} | {rec_lr:<20.4f}")
    print(f"{'F1-Score':<20} | {f1_dt:<25.4f} | {f1_lr:<20.4f}")
    print(f"{'ROC-AUC Score':<20} | {auc_dt:<25.4f} | {auc_lr:<20.4f}")
    print('=' * 70)
    
    # Confusion Matrix
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=ax[0], cbar=False,
                xticklabels=['Pred Low', 'Pred High'], yticklabels=['Actual Low', 'Actual High'])
    ax[0].set_title(f'Decision Tree Confusion Matrix\n(Accuracy: {acc_dt:.1%})', fontweight='bold')
    
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=ax[1], cbar=False,
                xticklabels=['Pred Low', 'Pred High'], yticklabels=['Actual Low', 'Actual High'])
    ax[1].set_title(f'Logistic Regression Confusion Matrix\n(Accuracy: {acc_lr:.1%})', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(img_dir, 'ml_confusion_matrices.png'), dpi=300)
    plt.close()
    
    # ROC Curves
    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.3f})', color='#3B82F6', linewidth=2.5)
    ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.3f})', color='#10B981', linewidth=2.5)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess Baseline (AUC = 0.500)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(os.path.join(img_dir, 'ml_roc_curves.png'), dpi=300)
    plt.close()
    
    # Feature Importance
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    feat_imp = pd.Series(dt_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    feat_imp = feat_imp[feat_imp > 0]
    ax[0].barh(feat_imp.index, feat_imp.values, color='#3B82F6', edgecolor='black', alpha=0.85)
    ax[0].set_title('Decision Tree Feature Importance', fontweight='bold')
    ax[0].set_xlabel('Gini Importance')
    for i, v in enumerate(feat_imp.values):
        ax[0].text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold')
        
    lr_coefs = pd.Series(lr_model.coef_[0], index=feature_cols).sort_values()
    colors_coef = ['#EF4444' if x < 0 else '#10B981' for x in lr_coefs.values]
    ax[1].barh(lr_coefs.index, lr_coefs.values, color=colors_coef, edgecolor='black', alpha=0.85)
    ax[1].set_title('Logistic Regression Standardized Coefficients', fontweight='bold')
    ax[1].set_xlabel('Coefficient Value')
    ax[1].axvline(0, color='black', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(os.path.join(img_dir, 'ml_feature_importance.png'), dpi=300)
    plt.close()
    
    print('\n[7/7] Successfully executed pipeline and generated all assets!')

if __name__ == '__main__':
    main()
