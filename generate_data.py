import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 1000

# Synthetic Sri Lankan Telecom/Broadband Features
tenure = np.random.randint(1, 72, size=n_samples)  # Months with provider
monthly_bill_lkr = np.random.uniform(1500.0, 12000.0, size=n_samples)  # Bill in LKR
support_tickets = np.random.poisson(lam=2, size=n_samples)  # Complaints / Help tickets
connection_type = np.random.choice(['Fibre', '4G Home Broadband', 'Postpaid Mobile'], size=n_samples, p=[0.40, 0.40, 0.20])
payment_method = np.random.choice(['eZ Cash / Frimi / e-Wallet', 'Bank Direct Debit', 'Over the Counter / Cash'], size=n_samples)

# Churn logic adapted for SL context
churn_score = (
    (support_tickets * 0.30) 
    + (monthly_bill_lkr / 10000) 
    - (tenure / 25) 
    + np.where(connection_type == '4G Home Broadband', 0.4, -0.3)
)
churn_probability = 1 / (1 + np.exp(-churn_score))
churn = (churn_probability > np.median(churn_probability)).astype(int)

df = pd.DataFrame({
    'Tenure_Months': tenure,
    'Monthly_Bill_LKR': monthly_bill_lkr,
    'Support_Tickets_30d': support_tickets,
    'Connection_Type': connection_type,
    'Payment_Method': payment_method,
    'Churn': churn
})

df.to_csv('churn_data.csv', index=False)
print("✅ Created 'churn_data.csv' with 1,000 Sri Lankan telecom records!")