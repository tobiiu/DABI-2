import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_tipping_periodicity(orders: pd.DataFrame, tips: pd.DataFrame):
    """
    Analyzes and visualizes tipping probability across different time periods
    (hour, day of week, month, season).
    """
    df = pd.merge(orders, tips, on='order_id', how='inner')
    df['hour'] = df['order_date'].dt.hour
    df['day_of_week'] = df['order_date'].dt.day_name()
    df['month'] = df['order_date'].dt.month
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=day_order, ordered=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Tipping Periodicity Analysis', fontsize=20)

    # Hourly Probability
    hourly_prob = df.groupby('hour')['tip'].mean() * 100
    sns.barplot(x=hourly_prob.index, y=hourly_prob.values, ax=axes[0, 0], palette='viridis', hue=hourly_prob.index)
    axes[0, 0].set_title('Tipping Probability by Hour of Day')
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Tip Probability (%)')

    # Daily Probability
    daily_prob = df.groupby('day_of_week')['tip'].mean() * 100
    sns.barplot(x=daily_prob.index, y=daily_prob.values, ax=axes[0, 1], palette='plasma', hue=daily_prob.index)
    axes[0, 1].set_title('Tipping Probability by Day of Week')
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Tip Probability (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Monthly Probability
    monthly_prob = df.groupby('month')['tip'].mean() * 100
    sns.barplot(x=monthly_prob.index, y=monthly_prob.values, ax=axes[1, 0], palette='magma', hue=monthly_prob.index)
    axes[1, 0].set_title('Tipping Probability by Month')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Tip Probability (%)')

    # Weekend vs Weekday
    df['is_weekend'] = df['order_date'].dt.dayofweek.isin([5, 6])
    weekend_prob = df.groupby('is_weekend')['tip'].mean() * 100
    sns.barplot(x=['Weekday', 'Weekend'], y=weekend_prob.values, ax=axes[1, 1], palette='cividis', hue=['Weekday', 'Weekend'])
    axes[1, 1].set_title('Tipping Probability: Weekday vs. Weekend')
    axes[1, 1].set_ylabel('Tip Probability (%)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('output/periodicity_analysis.png')
    #plt.show()
    print("Periodicity analysis plot saved to output/periodicity_analysis.png")


def plot_tipping_trend(orders: pd.DataFrame, tips: pd.DataFrame):
    """
    Analyzes and visualizes the long-term trend of tipping.
    Filters for weeks with a significant number of orders and fits a trend line.
    """
    df = pd.merge(orders, tips, on='order_id', how='inner')
    df['week'] = df['order_date'].dt.to_period('W')
    
    weekly_stats = df.groupby('week').agg(
        total_orders=('order_id', 'count'),
        tipped_orders=('tip', 'sum')
    ).reset_index()

    # Filter out weeks with low order volume for a stable trend
    stable_weeks = weekly_stats[weekly_stats['total_orders'] >= 1000].copy()
    stable_weeks['tip_percentage'] = (stable_weeks['tipped_orders'] / stable_weeks['total_orders']) * 100
    stable_weeks['time_index'] = range(len(stable_weeks))

    if stable_weeks.empty:
        print("Not enough data for trend analysis after filtering.")
        return

    # Define trend function (square root)
    def sqrt_trend(x, a, b):
        return a + b * np.sqrt(x)

    # Fit the trend
    x_data = stable_weeks['time_index']
    y_data = stable_weeks['tip_percentage']
    popt, _ = curve_fit(sqrt_trend, x_data, y_data)
    trend_line = sqrt_trend(x_data, *popt)

    # Plot
    plt.figure(figsize=(15, 7))
    plt.plot(stable_weeks['week'].astype(str), y_data, marker='o', linestyle='-', label='Weekly Tip %')
    plt.plot(stable_weeks['week'].astype(str), trend_line, color='red', linestyle='--', label=f'Square Root Trend (a={popt[0]:.2f}, b={popt[1]:.2f})')
    plt.title('Long-Term Trend of Tipping Percentage')
    plt.xlabel('Week')
    plt.ylabel('Tip Percentage (%)')
    plt.xticks(rotation=90)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/tipping_trend.png')
    #plt.show()
    print("Trend analysis plot saved to output/tipping_trend.png")