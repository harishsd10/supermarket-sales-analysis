# Executive Business Intelligence & Machine Learning Report
## Supermarket Sales Performance & Customer Satisfaction Modeling

**Dataset:** Kaggle Supermarket Sales (1,000 Records)  
**Branches:** Branch A (Yangon), Branch B (Mandalay), Branch C (Naypyitaw)  
**Period:** January 2019 - March 2019  

---

## 1. Executive Summary & Core KPIs
Over the recorded period, the supermarket chain generated **,966.75** in gross sales and **,379.37** in gross margin across **1,000 transactions**.

| Metric | Overall Value | Top Performer |
|---|---|---|
| **Total Revenue** | ,966.75 | Branch C (,568.71) |
| **Gross Margin Income** | ,379.37 | Branch C (,265.18) |
| **Total Quantity Sold** | 5,510 Units | Electronic Accessories (971 units) |
| **Average Order Value (AOV)** | .97 | Home & Lifestyle (.64) |
| **Customer Satisfaction (Rating)** | 6.97 / 10.0 | Branch A (7.03 / 10.0) |
| **High Satisfaction Ratio (Rating >= 7.0)** | 50.1% | Food & Beverages (53.4%) |

---

## 2. Exploratory Data Insights

### A. Product Line Performance
1. **Revenue Drivers**:
   - **Food & Beverages**: ,144.84 (17.4% of total sales)
   - **Sports & Travel**: ,122.83 (17.1% of total sales)
   - **Electronic Accessories**: ,337.53 (16.8% of total sales)
   - **Fashion Accessories**: ,305.90 (16.8% of total sales)
   - **Home & Lifestyle**: ,861.91 (16.7% of total sales)
   - **Health & Beauty**: ,193.74 (15.2% of total sales)
2. **Key Insight**: Revenue is remarkably balanced across all 6 product lines, indicating healthy multi-category customer interest without severe category dependencies.

### B. Branch & Geographic Comparison
- **Branch C (Naypyitaw)**: Highest total sales (.57K) and highest AOV (.10).
- **Branch A (Yangon)**: Highest customer rating (7.03 / 10.0) and highest transaction volume (340 orders).
- **Branch B (Mandalay)**: .20K revenue with 332 orders and average rating of 6.82 / 10.0.

### C. Customer Demographics & Behavior
- **Member vs. Normal**: Members contributed **,223.44** (50.85%) vs. Normal shoppers **,743.31** (49.15%).
- **High-Value Cohort**: **Female Members** represent the most lucrative customer segment with an Average Order Value of **.73** and total sales of **,146.94**.

### D. Payment Methods
- **E-wallet**: 34.5% market share (.99K revenue)
- **Cash**: 34.4% market share (.21K revenue)
- **Credit Card**: 31.1% market share (.77K revenue)

---

## 3. Supervised Machine Learning Benchmark

We built and evaluated two supervised classification models to predict whether a transaction will result in a **High Rating (>= 7.0)** or **Low Rating (< 7.0)**:

| Evaluation Metric | Decision Tree Classifier | Logistic Regression |
|---|---|---|
| **Accuracy** | **52.80%** | **53.20%** |
| **Precision** | **52.48%** | **53.12%** |
| **Recall** | **59.20%** | **54.40%** |
| **F1-Score** | **55.64%** | **53.75%** |
| **ROC-AUC Score** | **53.63%** | **51.83%** |

### Model Interpretability:
- **Decision Tree**: Key decision splits are driven by Unit price, Product line, Hour of transaction, and Tax 5%.
- **Logistic Regression**: Positive log-odds coefficients associated with Branch C, Quantity, and Unit Price.

---

## 4. Strategic Business Recommendations

1. **Implement Premium Cross-Category Bundling**:
   - Pair leading categories (*Food & Beverages* + *Sports & Travel*) into wellness bundle packs to raise basket size.
2. **Replicate Branch C Merchandising Across Branch A & B**:
   - Transfer Branch C's store navigation, shelf layouts, and high-ticket item placement to Branch A and B to raise their AOVs from  to +.
3. **Launch Digital Wallet Cashback Campaigns**:
   - Incentivize the 34.4% cash customers to switch to registered E-wallets/Member cards with 3-5% instant reward points to enable customer identity tracking.
4. **VIP Membership Loyalty Tiers**:
   - Design exclusive VIP perks for female members (our highest-spending demographic) with priority checkout and personalized beauty/fashion discounts.
5. **Staff Optimization for Peak Shopping Hours**:
   - Add express lanes and dynamic cashier staffing during peak hours (1:00 - 3:00 PM and 6:00 - 8:00 PM) to avoid delays that trigger low ratings.
