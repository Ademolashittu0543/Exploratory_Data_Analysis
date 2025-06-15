# Pizza Delivery EDA Analysis

## Overview
This repository contains an Exploratory Data Analysis (EDA) performed on a pizza delivery dataset. The analysis aims to uncover insights into delivery performance, focusing on factors such as delivery duration, distance, traffic impact, and peak hours. The notebook `EDA.ipynb` includes data exploration, statistical tests, and actionable recommendations.

## Dataset
- **Source**: The dataset includes 1004 rows and 25 columns with details on orders, such as `Order ID`, `Restaurant Name`, `Location`, `Order Time`, `Delivery Time`, `Delivery Duration (min)`, `Pizza Size`, `Pizza Type`, `Toppings Count`, `Distance (km)`, and more.
- **New Features**: 
  - `Delivery Speed (km/min)`: Calculated as `Distance (km) / Delivery Duration (min)`.
  - `Is Long Distance`: Boolean flag for orders with `Distance >= 5 km` (485 rows).

## Analysis Highlights
- **Statistical Tests**:
  - **T-test**: Significant difference in `Delivery Duration (min)` between peak and non-peak hours (t-stat=7.58, p-value=0.0000).
  - **ANOVA**: Significant variation in `Delivery Duration (min)` across traffic levels (F-stat=252.74, p-value=0.0000).
- **Insights**:
  - Peak hours and higher traffic levels significantly increase delivery times.
  - Long-distance deliveries (485 rows) suggest a need for optimized routing.

## Recommendations
- **Dynamic Resource Allocation**: Increase staff during peak hours (18:00-20:00) and adjust routes in high-traffic zones.
- **Peak Hour Adjustments**: Offer off-peak discounts and communicate wait times.
- **Traffic Mitigation**: Integrate real-time traffic data and partner with navigation apps.
- **Performance Monitoring**: Track durations by traffic and hour, test traffic-aware algorithms.
- **Long-Distance Optimization**: Improve speed for long-distance orders and balance resources.

## Files
- `EDA.ipynb`: Jupyter notebook with the full analysis, including data exploration, visualizations, and statistical tests.
- `Pizza_Delivery_EDA.pdf`: Compiled PDF version of the notebook (generated using the provided script).
- `EDA.py`: Re-executable code.
- `README.md`: This file.
- `seaborn`: Enhanced_pizza_sell_data_2024-25.xlsx

## Usage
1. Open `EDA.ipynb` in Google Colab or a local Jupyter environment.
2. Run all cells to reproduce the analysis.
3. Review the recommendations in the notebook's final section.

## Requirements
- Python 3.x
- Libraries: `pandas`, `numpy`, `scipy`, `plotly`,`seaborn`,`matplotlib`


