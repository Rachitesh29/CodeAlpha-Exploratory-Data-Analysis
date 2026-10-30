# CodeAlpha - Exploratory Data Analysis

## Project Overview
Exploratory Data Analysis of the Brazilian E-Commerce Public Dataset by Olist.

## Objectives
- Analyze e-commerce orders and customer behavior.
- Examine revenue and product category performance.
- Evaluate delivery performance and customer reviews.
- Identify missing values and duplicate records.
- Generate meaningful business insights.

## Technologies
- Python
- Pandas
- NumPy
- Matplotlib

## Dataset
Eight datasets were analyzed:
Customers, Geolocation, Orders, Order Items, Payments, Reviews, Products and Sellers.

Source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

## Key Results

| Metric | Result |
|---|---:|
| Total orders | 99,441 |
| Delivered orders | 96,478 |
| Cancelled orders | 625 |
| Total payment value | R$16,008,872.12 |
| Average payment per order | R$160.99 |
| Average review score | 4.09/5 |
| Average delivery time | 12.56 days |

## Key Findings
- Top revenue category: beleza_saude
- Most common payment method: credit_card
- Average customer review score: 4.09/5
- Geolocation dataset contained 261,831 duplicate rows.
- Total missing cells across datasets: 153,259.

## Visualizations
Nine charts were generated:
1. Monthly Orders
2. Monthly Revenue
3. Order Status
4. Category Revenue
5. Payment Types
6. Review Scores
7. Delivery Distribution
8. Customer States
9. Product Price vs Freight

## Project Structure
- data/processed - Cleaned analytical datasets
- reports/ - Business KPIs and EDA report
- reports/charts/ - Generated visualizations
- src/ - Python source code

## Execution

Install dependencies:
pip install -r requirements.txt

Run:
python -m src.eda

Note: Download the original Olist CSV datasets and place them in data/raw before running.

## Author
Rachitesh Sharma

CodeAlpha Data Analytics Internship
