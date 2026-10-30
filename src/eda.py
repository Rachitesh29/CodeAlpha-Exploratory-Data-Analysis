from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"
REPORTS = BASE / "reports"
CHARTS = REPORTS / "charts"

for folder in [PROCESSED, REPORTS, CHARTS]:
    folder.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

FILES = {
    "customers": "olist_customers_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv"
}

def load_data():
    data = {}
    for name, filename in FILES.items():
        path = RAW / filename
        data[name] = pd.read_csv(path)
        logging.info("%s loaded: %s rows, %s columns",
                     name, *data[name].shape)
    return data

def profile_data(data):
    profiles = []
    missing = []

    for name, df in data.items():
        profiles.append({
            "dataset": name,
            "rows": len(df),
            "columns": len(df.columns),
            "duplicates": int(df.duplicated().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum()/1048576, 2)
        })

        for column in df.columns:
            missing.append({
                "dataset": name,
                "column": column,
                "missing_count": int(df[column].isna().sum()),
                "missing_percent": round(df[column].isna().mean()*100, 2),
                "dtype": str(df[column].dtype)
            })

    pd.DataFrame(profiles).to_csv(
        PROCESSED / "dataset_profile.csv", index=False
    )
    pd.DataFrame(missing).to_csv(
        PROCESSED / "missing_values_report.csv", index=False
    )

    return profiles, missing

def prepare_data(data):
    orders = data["orders"].copy()
    items = data["items"].copy()
    payments = data["payments"].copy()
    reviews = data["reviews"].copy()
    products = data["products"].copy()
    customers = data["customers"].copy()
    sellers = data["sellers"].copy()

    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]

    for column in date_columns:
        if column in orders.columns:
            orders[column] = pd.to_datetime(
                orders[column], errors="coerce"
            )

    orders["purchase_month"] = (
        orders["order_purchase_timestamp"].dt.to_period("M").astype(str)
    )

    payment_summary = payments.groupby("order_id").agg(
        total_payment=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
        payment_methods=("payment_type", "nunique")
    ).reset_index()

    review_summary = reviews.groupby("order_id").agg(
        review_score=("review_score", "mean"),
        review_count=("review_score", "count")
    ).reset_index()

    order_analysis = orders.merge(
        customers[["customer_id", "customer_state"]],
        on="customer_id", how="left"
    )

    order_analysis = order_analysis.merge(
        payment_summary, on="order_id", how="left"
    )

    order_analysis = order_analysis.merge(
        review_summary, on="order_id", how="left"
    )

    if "order_delivered_customer_date" in orders.columns:
        order_analysis["delivery_days"] = (
            order_analysis["order_delivered_customer_date"] -
            order_analysis["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400

    order_analysis["delivery_delay_days"] = (
        order_analysis["order_delivered_customer_date"] -
        order_analysis["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    order_analysis.to_csv(
        PROCESSED / "order_analysis.csv", index=False
    )

    product_items = items.merge(
        products[["product_id", "product_category_name"]],
        on="product_id", how="left"
    )

    product_items = product_items.merge(
        sellers[["seller_id", "seller_state"]],
        on="seller_id", how="left"
    )

    product_items.to_csv(
        PROCESSED / "product_analysis.csv", index=False
    )

    return orders, items, payments, reviews, order_analysis, product_items

def save_chart(filename):
    plt.tight_layout()
    plt.savefig(CHARTS / filename, dpi=150, bbox_inches="tight")
    plt.close()

def create_visualizations(orders, items, payments, reviews,
                          order_analysis, product_items):

    # 1. Monthly order trend
    monthly = orders.groupby("purchase_month")["order_id"].nunique()
    monthly = monthly.sort_index()
    plt.figure(figsize=(12, 5))
    monthly.plot(marker="o")
    plt.title("Monthly Order Trend")
    plt.xlabel("Month")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=45)
    save_chart("monthly_orders.png")

    # 2. Monthly payment revenue
    revenue = order_analysis.groupby("purchase_month")[
        "total_payment"
    ].sum().dropna()
    plt.figure(figsize=(12, 5))
    revenue.plot(marker="o")
    plt.title("Monthly Payment Revenue")
    plt.xlabel("Month")
    plt.ylabel("Revenue (BRL)")
    plt.xticks(rotation=45)
    save_chart("monthly_revenue.png")

    # 3. Order status
    status = orders["order_status"].value_counts()
    plt.figure(figsize=(9, 5))
    status.plot(kind="bar")
    plt.title("Order Status Distribution")
    plt.xlabel("Order Status")
    plt.ylabel("Orders")
    plt.xticks(rotation=35)
    save_chart("order_status.png")

    # 4. Product category revenue
    category = product_items.groupby(
        "product_category_name"
    )["price"].sum().sort_values(ascending=False).head(15)

    plt.figure(figsize=(12, 7))
    category.sort_values().plot(kind="barh")
    plt.title("Top 15 Product Categories by Item Revenue")
    plt.xlabel("Revenue (BRL)")
    save_chart("category_revenue.png")

    # 5. Payment types
    payment_types = payments.groupby(
        "payment_type"
    )["payment_value"].sum().sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    payment_types.plot(kind="pie", autopct="%1.1f%%")
    plt.title("Revenue Share by Payment Type")
    plt.ylabel("")
    save_chart("payment_types.png")

    # 6. Review scores
    plt.figure(figsize=(9, 5))
    reviews["review_score"].value_counts().sort_index().plot(kind="bar")
    plt.title("Customer Review Score Distribution")
    plt.xlabel("Review Score")
    plt.ylabel("Number of Reviews")
    save_chart("review_scores.png")

    # 7. Delivery time
    delivery = order_analysis["delivery_days"].dropna()
    delivery = delivery[(delivery >= 0) & (delivery <= 100)]

    plt.figure(figsize=(9, 5))
    plt.hist(delivery, bins=30)
    plt.title("Order Delivery Time Distribution")
    plt.xlabel("Delivery Days")
    plt.ylabel("Frequency")
    save_chart("delivery_distribution.png")

    # 8. Customer geography
    state_orders = order_analysis.groupby(
        "customer_state"
    )["order_id"].nunique().sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    state_orders.sort_values().plot(kind="barh")
    plt.title("Top 15 States by Customer Orders")
    plt.xlabel("Orders")
    save_chart("customer_states.png")

    # 9. Price vs freight
    sample = product_items[["price", "freight_value"]].dropna()
    if len(sample) > 10000:
        sample = sample.sample(10000, random_state=42)

    plt.figure(figsize=(9, 6))
    plt.scatter(sample["price"], sample["freight_value"], alpha=0.3)
    plt.title("Product Price vs Freight Value")
    plt.xlabel("Product Price (BRL)")
    plt.ylabel("Freight Value (BRL)")
    save_chart("price_vs_freight.png")

def generate_insights(orders, payments, reviews,
                      order_analysis, product_items):

    total_orders = orders["order_id"].nunique()
    total_customers = orders["customer_id"].nunique()
    total_revenue = payments["payment_value"].sum()
    average_order_value = total_revenue / total_orders if total_orders else 0

    completed = orders["order_status"].eq("delivered").sum()
    cancelled = orders["order_status"].eq("canceled").sum()

    avg_review = reviews["review_score"].mean()
    avg_delivery = order_analysis["delivery_days"].dropna().mean()

    top_category = (
        product_items.groupby("product_category_name")["price"]
        .sum().idxmax()
    )

    top_payment = payments["payment_type"].value_counts().idxmax()

    insights = [
        "OLIST E-COMMERCE - EXPLORATORY DATA ANALYSIS",
        "=" * 60,
        "",
        "BUSINESS OVERVIEW",
        f"Total orders: {total_orders:,}",
        f"Unique customers: {total_customers:,}",
        f"Total payment value: BRL {total_revenue:,.2f}",
        f"Average payment value per order: BRL {average_order_value:,.2f}",
        f"Delivered orders: {completed:,}",
        f"Cancelled orders: {cancelled:,}",
        f"Average review score: {avg_review:.2f}/5",
        f"Average delivery time: {avg_delivery:.2f} days",
        "",
        "KEY FINDINGS",
        f"Top category by item revenue: {top_category}",
        f"Most common payment method: {top_payment}",
        "",
        "INTERPRETATION",
        "- Order status helps identify operational performance.",
        "- Monthly revenue shows business growth patterns.",
        "- Product categories highlight sales concentration.",
        "- Review scores provide customer satisfaction signals.",
        "- Delivery time helps evaluate logistics performance.",
        "",
        "NOTE",
        "Revenue uses recorded payment values. Category revenue uses item prices.",
        "The two measures have different definitions."
    ]

    (REPORTS / "eda_report.txt").write_text(
        "\n".join(insights), encoding="utf-8"
    )

    pd.DataFrame([{
        "total_orders": total_orders,
        "unique_customers": total_customers,
        "total_payment_value": total_revenue,
        "average_payment_per_order": average_order_value,
        "delivered_orders": completed,
        "cancelled_orders": cancelled,
        "average_review_score": avg_review,
        "average_delivery_days": avg_delivery
    }]).to_csv(REPORTS / "business_kpis.csv", index=False)

    print("\n".join(insights))

def main():
    print("\n" + "=" * 60)
    print("CODEALPHA TASK 2 - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    data = load_data()

    profiles, missing = profile_data(data)

    print("\nDATASET PROFILE")
    print(pd.DataFrame(profiles).to_string(index=False))

    print("\nDATA QUALITY SUMMARY")
    print(f"Total datasets: {len(data)}")
    print(f"Total missing cells: {sum(x['missing_count'] for x in missing):,}")

    orders, items, payments, reviews, order_analysis, product_items = (
        prepare_data(data)
    )

    create_visualizations(
        orders, items, payments, reviews, order_analysis, product_items
    )

    generate_insights(
        orders, payments, reviews, order_analysis, product_items
    )

    print("\nEDA COMPLETED SUCCESSFULLY!")
    print(f"Charts: {CHARTS}")
    print(f"Reports: {REPORTS}")
    print(f"Processed data: {PROCESSED}")

if __name__ == "__main__":
    main()
