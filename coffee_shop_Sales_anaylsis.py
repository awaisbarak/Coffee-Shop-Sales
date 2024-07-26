import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

file_path = r'C:\Users\Zenbook\PycharmProjects\pythonProject\Coffee Shop Sales.xlsx'

xls = pd.ExcelFile(file_path)
sheets = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}

sheets['Sheet6'].dropna(inplace=True)
data = sheets[xls.sheet_names[0]]

data['transaction_date'] = pd.to_datetime(data['transaction_date'])
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
data['total_price'] = data['transaction_qty'] * data['unit_price']

category_sales = data.groupby('product_category')['total_price'].sum().reset_index()

#plotting bar chart of Total Sales by Product Category
plt.figure(figsize=(10, 6))
plt.bar(category_sales['product_category'], category_sales['total_price'], color='skyblue')
plt.title('Total Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

daily_profit_loss = data.groupby('transaction_date')['total_price'].sum().reset_index()

daily_profit_loss['cumulative_profit_loss'] = daily_profit_loss['total_price'].cumsum()

# Plot cumulative profit/loss over time (Line Chart)
plt.figure(figsize=(12, 6))
plt.plot(daily_profit_loss['transaction_date'], daily_profit_loss['cumulative_profit_loss'], label='Cumulative Profit/Loss')
plt.xlabel('Date')
plt.ylabel('Cumulative Profit/Loss')
plt.title('Cumulative Profit/Loss Over Time')
plt.legend()
plt.tight_layout()
plt.show()

# Profit/Loss Analysis - Group by product to calculate total sales and profit
product_sales = data.groupby(['product_category', 'product_type']).agg({
    'transaction_qty': 'sum',
    'unit_price': 'mean'
}).reset_index()

product_sales['total_sales'] = product_sales['transaction_qty'] * product_sales['unit_price']

profitable_products = product_sales[product_sales['total_sales'] > product_sales['total_sales'].median()]

loss_making_products = product_sales[product_sales['total_sales'] <= product_sales['total_sales'].median()]

# Predict future profits  using linear regression
data['month'] = data['transaction_date'].dt.month
monthly_sales = data.groupby('month')['transaction_qty'].sum().reset_index()

# Splitting data into training and testing sets
X = monthly_sales[['month']]
y = monthly_sales['transaction_qty']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

future_months = pd.DataFrame({'month': [7, 8, 9, 10, 11, 12]})
predictions = model.predict(future_months)

# Plot future sales predictions (Line Chart)
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales['month'], monthly_sales['transaction_qty'], label='Actual Sales')
plt.plot(future_months['month'], predictions, label='Predicted Sales', linestyle='--')
plt.xlabel('Month')
plt.ylabel('Sales Quantity')
plt.title('Sales Predictions for Future Months')
plt.legend()
plt.tight_layout()
plt.show()

loss_making_analysis = data[data['product_type'].isin(loss_making_products['product_type'])]

plt.figure(figsize=(12, 6))
for product in loss_making_products['product_type']:
    product_data = loss_making_analysis[loss_making_analysis['product_type'] == product]
    product_sales_over_time = product_data.groupby('transaction_date')['total_price'].sum().reset_index()
    plt.plot(product_sales_over_time['transaction_date'], product_sales_over_time['total_price'], label=product)


plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.title('Sales Trend of Loss-Making Products Over Time')
plt.legend()
plt.tight_layout()
plt.show()

# Save cleaned data to a new Excel file
with pd.ExcelWriter('Cleaned_Coffee_Shop_Sales.xlsx') as writer:
    data.to_excel(writer, sheet_name='Main Data', index=False)
    sheets['Sheet6'].to_excel(writer, sheet_name='Sheet6_Cleaned', index=False)
