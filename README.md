# Customer Segmentation Project

## Introduction

This project is my submission for the technical test at Upscale for the Data Scientist position. The main objective of this project is to perform customer segmentation for a supermarket based on various demographic and purchasing behavior factors. The goal is to divide customers into distinct groups to create separate marketing and sales strategies for each segment.

## Data Source

The dataset used for this project was obtained from [Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis). It contains information about customers' demographics, shopping preferences, and purchase history.

## Process Overview

The customer segmentation process follows these steps:

1. Data Cleaning: The dataset is preprocessed to handle missing values and ensure data quality.

2. Feature Engineering: Various features are created from the available data to capture meaningful information.

3. Encoding: Categorical features are encoded to numerical values for modeling.

4. Scaling: The numerical features are scaled to ensure they have a similar impact on the segmentation model.

5. K-means Clustering: The K-means algorithm is used to cluster customers into distinct groups.

6. Segment Interpretation: The clusters are interpreted to understand the characteristics of each customer segment.

## Segmentation Results

Based on the K-means clustering, the following customer segments were identified:

1. Young Families:
   - Mainly consists of young couples with lower incomes.
   - Likely young families with growing responsibilities.
   - Diverse preferences, purchasing wines, meat, and gold products.
   - Prefer shopping in physical stores and tend to choose cheaper products.

2. Premium Seniors:
   - Majority are couples with high incomes.
   - Wide age range, including retirees.
   - They do not have children and show a preference for expensive products.
   - Shop using various methods, including online, catalogs, and offline stores.

3. Established Households:
   - Mostly composed of families with moderate incomes.
   - In their middle-aged years, indicating established families with children.
   - Focus on purchasing wines and prefer both online and in-store shopping.
   - Mindful of costs and make significant purchases of affordable products.

4. Individual Shoppers:
   - Consists of individuals without partners and with lower incomes.
   - In their middle-aged years, representing various life stages and experiences.
   - Diverse shopping patterns, buying wines, meat, and gold items.
   - Flexible with shopping methods, enjoying both online and in-store shopping.

## Marketing and Sales Strategies

For each customer segment, the following marketing and sales strategies are recommended:

**Young Families:**
- Emphasize family-oriented messaging and affordable product options.
- Utilize social media and targeted online ads to reach this tech-savvy group.
- Offer in-store exclusive deals and create a warm shopping atmosphere for families.
- Collaborate with parenting blogs and influencers to promote family-friendly products.

**Premium Seniors:**
- Position products as premium and high-quality to appeal to refined tastes.
- Use sophisticated branding and leverage online advertising, catalogs, and in-store displays.
- Create exclusive membership programs with personalized offers.
- Collaborate with luxury lifestyle influencers and celebrities to endorse the brand.

**Established Households:**
- Highlight wines and offer targeted online ads and email campaigns.
- Showcase affordable products and offer family-sized packages.
- Collaborate with food and lifestyle bloggers to promote family-friendly items.

**Individual Shoppers:**
- Appeal to their independence and flexibility through targeted online advertising.
- Offer a diverse product range and online-exclusive deals.
- Collaborate with lifestyle bloggers to showcase versatile products.

## Conclusion

The customer segmentation project has successfully identified distinct customer segments based on demographics and purchasing behavior. With the insights gained, tailored marketing and sales strategies can be implemented to meet the specific needs and preferences of each segment.
