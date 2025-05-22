# Import Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import lightgbm as lgb
import folium
import webbrowser
from folium.plugins import HeatMap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

def main():
    print("\n" + "="*50)
    print("ENHANCED ELECTRIC VEHICLE POPULATION ANALYSIS")
    print("="*50)
    
    # Load the Dataset
    file_path = 'Electric_Vehicle_Population_Data.csv'
    evp_data = pd.read_csv(file_path)

    # Improved Data Quality Analysis
    def analyze_data_quality(df):
        """Analyze and print data quality metrics"""
        missing_vals = df.isnull().sum()
        missing_percent = (missing_vals / len(df)) * 100
        
        # Find columns with zeros
        num_cols = df.select_dtypes(include=['number']).columns
        zero_counts = {col: (df[col] == 0).sum() for col in num_cols}
        
        print("\nColumns with missing values:")
        print(missing_vals[missing_vals > 0])
        
        print("\nColumns with zeros:")
        for col, count in zero_counts.items():
            if count > 0:
                print(f"{col}: {count} zeros ({count/len(df)*100:.2f}%)")
        
        return zero_counts
    
    # Run data quality analysis
    zero_counts = analyze_data_quality(evp_data)
    
    # Improved Data Cleaning - Better zero handling
    # FEATURE 1: Improved domain-specific data cleaning
    fill_dict = {
        'County': "Unknown",
        'City': "Unknown",
        'Postal Code': "Unknown",
        'Electric Utility': "Unknown",
        '2020 Census Tract': "Unknown",
        'Legislative District': evp_data['Legislative District'].mode()[0] if not evp_data['Legislative District'].isna().all() else "Unknown",
        'Vehicle Location': "Unknown"
    }
    
    # Handle zeros differently based on column type
    if 'Electric Range' in evp_data.columns:
        # For Electric Range, zeros are invalid - replace with vehicle-type specific median
        evp_data['Electric Range'] = evp_data['Electric Range'].replace(0, np.nan)
        
        # Impute based on vehicle type when possible
        if 'Electric Vehicle Type' in evp_data.columns:
            for vtype in evp_data['Electric Vehicle Type'].unique():
                type_median = evp_data[
                    (evp_data['Electric Vehicle Type'] == vtype) & 
                    (evp_data['Electric Range'].notna())
                ]['Electric Range'].median()
                
                evp_data.loc[
                    (evp_data['Electric Vehicle Type'] == vtype) & 
                    (evp_data['Electric Range'].isna()),
                    'Electric Range'
                ] = type_median
        
        # Fill any remaining NaN with overall median
        evp_data['Electric Range'] = evp_data['Electric Range'].fillna(evp_data['Electric Range'].median())
    
    if 'Base MSRP' in evp_data.columns:
        # For Base MSRP, check if mean is very low (potential data error)
        msrp_mean = evp_data['Base MSRP'].mean()
        
        # If mean MSRP is suspiciously low (under $1000), it might need scaling
        if msrp_mean < 1000 and msrp_mean > 0:
            print(f"Warning: Mean MSRP is ${msrp_mean:.2f}. Scaling prices...")
            evp_data['Base MSRP'] = evp_data['Base MSRP'] * 1000
        
        # Then handle zeros
        evp_data['Base MSRP'] = evp_data['Base MSRP'].replace(0, np.nan)
        
        # Impute based on make when possible
        if 'Make' in evp_data.columns:
            for make in evp_data['Make'].unique():
                make_median = evp_data[
                    (evp_data['Make'] == make) & 
                    (evp_data['Base MSRP'].notna())
                ]['Base MSRP'].median()
                
                evp_data.loc[
                    (evp_data['Make'] == make) & 
                    (evp_data['Base MSRP'].isna()),
                    'Base MSRP'
                ] = make_median
        
        # Fill any remaining NaN with overall median
        evp_data['Base MSRP'] = evp_data['Base MSRP'].fillna(evp_data['Base MSRP'].median())
    
    # Fill remaining NaN values
    evp_data = evp_data.fillna(fill_dict)
    
    # Feature Engineering: Extract Coordinates
    def extract_coordinates(point):
        if isinstance(point, str) and point.startswith("POINT"):
            coords = point.strip("POINT ()").split()
            if len(coords) == 2:
                return float(coords[1]), float(coords[0])
        return (np.nan, np.nan)
    
    evp_data['Latitude'], evp_data['Longitude'] = zip(*evp_data['Vehicle Location'].apply(extract_coordinates))

    # Improved coordinate handling
    zero_coords = (evp_data['Latitude'] == 0) | (evp_data['Longitude'] == 0)
    evp_data.loc[zero_coords, ['Latitude', 'Longitude']] = np.nan
    
    # Additional Data Cleaning
    # Calculate current year dynamically
    import datetime
    current_year = datetime.datetime.now().year
    
    # Calculate Vehicle Age
    evp_data['Vehicle Age'] = current_year - evp_data['Model Year']
    
    # FEATURE 2: Efficiency Metrics - Added Price per Range Mile and Efficiency categories
    evp_data['Price per Mile'] = evp_data['Base MSRP'] / evp_data['Electric Range']
    evp_data['Price per Mile'] = evp_data['Price per Mile'].replace([np.inf, -np.inf], np.nan)
    evp_data['Price per Mile'] = evp_data['Price per Mile'].fillna(evp_data['Price per Mile'].median())
    
    # Calculate efficiency rating (lower price per mile is better)
    price_mile_quantiles = evp_data['Price per Mile'].quantile([0.25, 0.5, 0.75]).tolist()
    
    def get_efficiency_category(price_per_mile):
        if price_per_mile <= price_mile_quantiles[0]:
            return 'Excellent'
        elif price_per_mile <= price_mile_quantiles[1]:
            return 'Good'
        elif price_per_mile <= price_mile_quantiles[2]:
            return 'Average'
        else:
            return 'Below Average'
    
    evp_data['Efficiency Rating'] = evp_data['Price per Mile'].apply(get_efficiency_category)
    
    # Group by County and calculate summary statistics
    country_summary = evp_data.groupby('County').agg({
        'Model': 'count',                # Count of vehicles
        'Electric Range': 'mean',        # Average electric range
        'Base MSRP': 'mean'              # Average price
    }).rename(columns={'Model': 'Vehicle Count'})
    country_summary = country_summary.sort_values(by='Vehicle Count', ascending=False)
    print(country_summary.head())

    # Set plot style
    sns.set_style("whitegrid")

    # Top 10 EV Manufacturers
    plt.figure(figsize=(10, 4))
    top_makes = evp_data["Make"].value_counts().nlargest(10)
    sns.barplot(x=top_makes.index, y=top_makes.values, hue=top_makes.index, palette="viridis", legend=False)
    plt.xlabel("Manufacturer")
    plt.ylabel("Count")
    plt.title("Top 10 EV Manufacturers")
    plt.xticks(rotation=45)
    sns.despine()
    plt.tight_layout()
    plt.savefig("bar_EV_Manufactures.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Top 10 EV Cars
    plt.figure(figsize=(10, 4))
    top_cars = evp_data["Model"].value_counts().nlargest(10)
    sns.barplot(x=top_cars.index, y=top_cars.values, hue=top_cars.index, palette="viridis", legend=False)
    plt.xlabel("Cars")
    plt.ylabel("Count")
    plt.title("Top 10 EV Cars")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("bar_EV_Cars.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Pie chart for Electric Vehicle Type
    plt.figure(figsize=(8, 8))
    evp_type_counts = evp_data['Electric Vehicle Type'].value_counts()
    plt.pie(evp_type_counts, labels=evp_type_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen'])
    plt.title("Distribution of Electric Vehicle Types")
    plt.tight_layout()
    plt.savefig("pie_evt.png", dpi=300)
    plt.close()
    
    # Vehicle Age Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(evp_data["Vehicle Age"], bins=20, kde=True, color="purple")
    plt.xlabel("Vehicle Age (Years)")
    plt.ylabel("Count")
    plt.title("Vehicle Age Distribution")
    plt.tight_layout()
    plt.savefig("Vehicle_Age_Dist.png", dpi=300)
    plt.close()
    
    # Electric Range Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(evp_data["Electric Range"], bins=30, kde=True, color="green")
    plt.xlabel("Electric Range (Miles)")
    plt.ylabel("Count")
    plt.title("Electric Range Distribution")
    plt.tight_layout()
    plt.savefig("Electric_Range_Dist.png", dpi=300)
    plt.close()
    
    # IMPROVED BOXPLOT: Enhanced Electric Range Boxplot by EV Type
    plt.figure(figsize=(12, 8))
    # Create an enhanced violinplot with overlaid boxplot for better visibility
    ax = sns.violinplot(x="Electric Vehicle Type", y="Electric Range", data=evp_data, 
                        palette="coolwarm", inner=None, alpha=0.4)
    sns.boxplot(x="Electric Vehicle Type", y="Electric Range", data=evp_data, 
               palette="coolwarm", width=0.3, ax=ax)
    
    # Add individual points for transparency using stripplot (with reduced opacity)
    sns.stripplot(x="Electric Vehicle Type", y="Electric Range", data=evp_data,
                 size=3, color=".3", alpha=0.2, ax=ax)
    
    # Add mean markers with different shape
    means = evp_data.groupby('Electric Vehicle Type')['Electric Range'].mean()
    plt.scatter(range(len(means)), means, color='white', s=100, marker='D', 
               edgecolor='black', zorder=3, label='Mean')
    
    # Add median labels
    medians = evp_data.groupby('Electric Vehicle Type')['Electric Range'].median()
    for i, median in enumerate(medians):
        plt.text(i, median + 5, f'Median: {median:.0f}', ha='center', va='bottom', 
                fontweight='bold', color='black', backgroundcolor='white', alpha=0.8)
    
    # Improve styling
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xlabel("Electric Vehicle Type", fontsize=14)
    plt.ylabel("Electric Range (Miles)", fontsize=14)
    plt.title("Electric Range Distribution by EV Type", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper right')
    
    # Add range annotations
    for i, ev_type in enumerate(evp_data['Electric Vehicle Type'].unique()):
        type_data = evp_data[evp_data['Electric Vehicle Type'] == ev_type]['Electric Range']
        plt.annotate(f'Range: {type_data.min():.0f}-{type_data.max():.0f} miles',
                    xy=(i, type_data.max()), xytext=(i, type_data.max() + 20),
                    ha='center', va='bottom', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig("enhanced_boxplot.png", dpi=300)
    plt.close()
    
    # NEW VISUALIZATION: Efficiency analysis by manufacturer
    plt.figure(figsize=(12, 6))
    # Get top 10 manufacturers by count
    top_10_makes = evp_data['Make'].value_counts().nlargest(10).index
    
    # Filter for these manufacturers
    efficiency_data = evp_data[evp_data['Make'].isin(top_10_makes)]
    
    # Create barplot with Efficiency Rating as hue
    sns.barplot(x='Make', y='Price per Mile', data=efficiency_data, hue='Efficiency Rating', 
                palette={'Excellent': 'green', 'Good': 'lightgreen', 
                         'Average': 'orange', 'Below Average': 'red'})
    
    plt.title('Price per Mile Efficiency by Manufacturer', fontsize=16)
    plt.xlabel('Manufacturer', fontsize=14)
    plt.ylabel('Price per Mile ($)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend(title='Efficiency Rating')
    plt.tight_layout()
    plt.savefig("price_per_mile_efficiency.png", dpi=300)
    plt.close()
    
    # Vehicle Age vs. Electric Range Scatter Plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=evp_data["Vehicle Age"], y=evp_data["Electric Range"], alpha=0.5, color="blue")
    plt.xlabel("Vehicle Age (Years)")
    plt.ylabel("Electric Range (Miles)")
    plt.title("Vehicle Age vs. Electric Range")
    plt.tight_layout()
    plt.savefig("va_er.png", dpi=300)
    plt.close()
    
    # Outlier Detection using IQR - More efficient implementation
    print("\nOutlier Analysis:")
    for col in ["Electric Range", "Base MSRP", "Vehicle Age", "Price per Mile"]:
        Q1 = evp_data[col].quantile(0.25)
        Q3 = evp_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_count = ((evp_data[col] < lower_bound) | (evp_data[col] > upper_bound)).sum()
        outliers_pct = outliers_count / len(evp_data) * 100
        
        print(f"Outliers in {col}: {outliers_count:,} records ({outliers_pct:.2f}%)")
        print(f"  Range: {lower_bound:.2f} to {upper_bound:.2f}")
    
    # Modeling Preparation
    print("\nPreparing for predictive modeling...")
    features = ['Electric Vehicle Type', 'Make', 'Model', 'Model Year', 'Base MSRP', 'Vehicle Age']
    target = 'Electric Range'
    
    # Encode categorical variables more efficiently
    le_dict = {}
    categorical_cols = ['Electric Vehicle Type', 'Make', 'Model']
    
    for col in categorical_cols:
        le = LabelEncoder()
        evp_data[col + '_Encoded'] = le.fit_transform(evp_data[col].astype(str))
        le_dict[col] = le  # Store for future reference

    # Modeling Preparation
    features = ['Electric Vehicle Type_Encoded', 'Make_Encoded', 'Model_Encoded', 'Model Year', 'Base MSRP', 'Vehicle Age']
    
    X = evp_data[features]
    y = evp_data[target]
    
    # Check for NaN values in target
    if y.isnull().sum() > 0:
        print(f"Warning: Target contains {y.isnull().sum()} NaN values. Dropping these rows.")
        valid_idx = y.notnull()
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
    
    # Split data with a fixed random seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Decision Tree Regressor with simpler hyperparameter tuning
    print("\nTraining Decision Tree Regressor...")
    start_time = time.time()
    
    # Simplified parameter grid for faster execution
    param_grid = {
        'max_depth': [5, 8],
        'min_samples_split': [5, 10]
    }
    
    regressor = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(regressor, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_dt = grid_search.best_estimator_
    DTR_pred = best_dt.predict(X_test)
    
    mse = mean_squared_error(y_test, DTR_pred)
    mae = mean_absolute_error(y_test, DTR_pred)
    r2 = r2_score(y_test, DTR_pred)
    execution_time = time.time() - start_time
    
    print("Decision Tree Regressor Performance:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.4f}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Feature Importance Plot for Decision Tree
    feature_importance = best_dt.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = feature_importance[sorted_idx]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=sorted_importance, y=sorted_features)
    plt.title("Feature Importance for Electric Range", fontsize=16)
    plt.xlabel("Importance Score", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.tight_layout()
    plt.savefig("feature_importance_decision_tree.png", dpi=300)
    plt.close()
    
    # Simplified decision tree visualization
    plt.figure(figsize=(20, 10))
    plot_tree(best_dt, max_depth=3, feature_names=features, filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree Plot (Limited to Depth 3)", fontsize=16)
    plt.tight_layout()
    plt.savefig("decision_tree_electric_range.png", dpi=300)
    plt.close()
    
    # Save decision tree rules to a text file
    tree_rules = export_text(best_dt, feature_names=features)
    with open("decision_tree_electric_range_rules.txt", "w") as f:
        f.write(tree_rules)
    
    # Scatter Plot: Actual vs Predicted (Decision Tree)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, DTR_pred, alpha=0.6, color="blue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title("Actual vs Predicted Electric Range (Decision Tree)", fontsize=16)
    plt.xlabel("Actual Electric Range", fontsize=14)
    plt.ylabel("Predicted Electric Range", fontsize=14)
    plt.tight_layout()
    plt.savefig("actual_vs_predicted_decision_tree.png", dpi=300)
    plt.close()
    
    # LightGBM Regressor - Optimized settings
    print("\nTraining LightGBM Regressor...")
    start_time = time.time()
    
    # Use default hyperparameters with conservative settings for better stability
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=8,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    
    lgb_mae = mean_absolute_error(y_test, lgb_pred)
    lgb_mse = mean_squared_error(y_test, lgb_pred)
    lgb_r2 = r2_score(y_test, lgb_pred)
    lgb_time = time.time() - start_time
    
    print("\nLightGBM Regressor Performance:")
    print(f"Mean Absolute Error: {lgb_mae:.2f}")
    print(f"Mean Squared Error: {lgb_mse:.2f}")
    print(f"R-squared: {lgb_r2:.4f}")
    print(f"Execution Time: {lgb_time:.2f} seconds")
    
    # LightGBM Feature Importance Plot
    lgb_feature_importance = lgb_model.feature_importances_
    sorted_idx_lgb = np.argsort(lgb_feature_importance)[::-1]
    sorted_features_lgb = [features[i] for i in sorted_idx_lgb]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=lgb_feature_importance[sorted_idx_lgb], y=sorted_features_lgb)
    plt.title("LightGBM Feature Importance", fontsize=16)
    plt.xlabel("Importance Score", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.tight_layout()
    plt.savefig("feature_importance_lightGBM.png", dpi=300)
    plt.close()
    
    # Scatter Plot: Actual vs Predicted (LightGBM)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, lgb_pred, alpha=0.6, color="blue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title("Actual vs Predicted Electric Range (LightGBM)", fontsize=16)
    plt.xlabel("Actual Electric Range", fontsize=14)
    plt.ylabel("Predicted Electric Range", fontsize=14)
    plt.tight_layout()
    plt.savefig("actual_vs_predicted_lightGBM.png", dpi=300)
    plt.close()
    
    # Time Series Analysis: EV Adoption Trend
    ev_adoption_trend = evp_data['Model Year'].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    plt.plot(ev_adoption_trend.index, ev_adoption_trend.values, marker='o', linestyle='-')
    plt.xlabel("Model Year")
    plt.ylabel("Number of EV Registrations")
    plt.title("Trend of EV Adoption Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ev_adoption.png", dpi=300)
    plt.close()
    
    # Simple Geospatial Analysis - Static map instead of interactive to avoid errors
    print("\nGenerating geospatial visualization...")
    if {'Latitude', 'Longitude'}.issubset(evp_data.columns):
        # Filter valid coordinates
        df_geo = evp_data.dropna(subset=['Latitude', 'Longitude']).copy()
        valid_coords = (
            df_geo['Latitude'].between(-90, 90) & 
            df_geo['Longitude'].between(-180, 180)
        )
        df_geo = df_geo[valid_coords]
        
        if not df_geo.empty:
            plt.figure(figsize=(12, 10))
            plt.scatter(
                df_geo['Longitude'], 
                df_geo['Latitude'],
                alpha=0.5,
                s=5,
                c=df_geo['Electric Vehicle Type'].astype('category').cat.codes,
                cmap='viridis'
            )
            plt.title("EV Distribution by Location", fontsize=16)
            plt.xlabel("Longitude", fontsize=14)
            plt.ylabel("Latitude", fontsize=14)
            plt.colorbar(label="Vehicle Type")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig("ev_geo_distribution.png", dpi=300)
            plt.close()
        else:
            print("No valid geospatial data available")
    else:
        print("Latitude/Longitude columns missing - check data processing")
    
    # Correlation Analysis: Factors Affecting EV Range
    num_corr_cols = ['Electric Range', 'Base MSRP', 'Model Year', 'Vehicle Age', 'Price per Mile']
    df_corr = evp_data[num_corr_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap of EV Factors", fontsize=16)
    plt.tight_layout()
    plt.savefig("cor_heatmap.png", dpi=300)
    plt.close()
    
    # NEW VISUALIZATION: Efficiency distribution by vehicle type
    plt.figure(figsize=(12, 6))
    efficiency_counts = pd.crosstab(
        evp_data['Electric Vehicle Type'], 
        evp_data['Efficiency Rating']
    )
    
    # Convert to percentage
    efficiency_pct = efficiency_counts.div(efficiency_counts.sum(axis=1), axis=0) * 100
    
    # Plot
    efficiency_pct.plot(kind='bar', stacked=True, 
                       colormap='RdYlGn_r')  # Red-Yellow-Green reversed (Red=Bad, Green=Good)
    
    plt.title('Efficiency Rating Distribution by Vehicle Type', fontsize=16)
    plt.xlabel('Vehicle Type', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend(title='Efficiency Rating')
    plt.tight_layout()
    plt.savefig("efficiency_by_vehicle_type.png", dpi=300)
    plt.close()
    
    # Model comparison summary
    print("\nModel Performance Comparison:")
    print("-" * 50)
    print(f"{'Metric':<20} | {'Decision Tree':<15} | {'LightGBM':<15}")
    print("-" * 50)
    print(f"{'MAE':<20} | {mae:<15.2f} | {lgb_mae:<15.2f}")
    print(f"{'MSE':<20} | {mse:<15.2f} | {lgb_mse:<15.2f}")
    print(f"{'R-squared':<20} | {r2:<15.4f} | {lgb_r2:<15.4f}")
    print(f"{'Execution Time (s)':<20} | {execution_time:<15.2f} | {lgb_time:<15.2f}")
    
    # Create a summary report
    print("\nAnalysis complete! Summary of findings:")
    print(f"- Dataset contains {len(evp_data):,} electric vehicles")
    print(f"- Top manufacturer: {evp_data['Make'].value_counts().index[0]}")
    print(f"- Most efficient EV type: {efficiency_pct['Excellent'].idxmax()}")
    print(f"- Best predictor of range: {sorted_features[0]}")
    print(f"- Best model: {'LightGBM' if lgb_r2 > r2 else 'Decision Tree'} with R² = {max(r2, lgb_r2):.4f}")
    print("\nAll visualizations have been saved as PNG files.")

if __name__ == '__main__':
    main()