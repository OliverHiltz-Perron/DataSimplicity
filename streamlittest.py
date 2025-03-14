import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import StringIO

st.set_page_config(layout="wide", page_title="Data Explorer Dashboard")

# Title and description
st.title("📊 Interactive Data Explorer")
st.markdown("""
Upload your CSV or Excel file to get instant visualizations and statistics. 
No coding required!
""")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])

# Function to determine column types
def get_column_types(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    return numeric_cols, categorical_cols, date_cols

# Function to generate summary statistics
def generate_summary(df):
    summary = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null %': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    
    # Add numeric statistics where applicable
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in df.columns:
        if col in numeric_cols:
            summary.loc[summary['Column'] == col, 'Min'] = df[col].min()
            summary.loc[summary['Column'] == col, 'Max'] = df[col].max()
            summary.loc[summary['Column'] == col, 'Mean'] = df[col].mean().round(2)
            summary.loc[summary['Column'] == col, 'Median'] = df[col].median()
            summary.loc[summary['Column'] == col, 'Std Dev'] = df[col].std().round(2)
    
    return summary

# Move the main visualization logic into a function
def analyze_data(df):
    # Display basic info
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10))
    
    # Display shape
    row_count, col_count = df.shape
    st.write(f"📏 Dimensions: {row_count} rows × {col_count} columns")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary Stats", "Data Cleaning", "Visualizations", "Correlations", "Data Explorer"])
    
    with tab1:
        st.subheader("📊 Summary Statistics")
        summary = generate_summary(df)
        st.dataframe(summary, use_container_width=True)
        
        # Additional summary for numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            st.subheader("🔢 Numeric Columns - Detailed Statistics")
            st.dataframe(df[numeric_cols].describe().T)
    with tab2:
        st.subheader("🧹 Data Cleaning")
        st.write("This section allows you to clean your data by handling missing values, duplicates, and outliers.")
        
        # Handle missing values
        st.subheader("🔍 Missing Values")
        
        # Initialize session state for tracking dropped columns
        if 'dropped_columns_history' not in st.session_state:
            st.session_state.dropped_columns_history = []
        if 'original_data' not in st.session_state:
            st.session_state.original_data = df.copy()
        
        # Find columns with missing values
        missing_values = df.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        
        if not columns_with_missing.empty:
            st.write("The following columns contain missing values:")
            
            # Create a DataFrame with missing value information
            missing_df = pd.DataFrame({
                'Column': columns_with_missing.index,
                'Missing Values': columns_with_missing.values,
                'Missing %': (columns_with_missing / len(df) * 100).round(2)
            })
            
            # Display the missing values table
            st.dataframe(missing_df, use_container_width=True)
            
            # Add drop buttons for each column
            st.write("Select columns to drop rows with missing values:")
            
            for col in columns_with_missing.index:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{col}**: {missing_values[col]} missing values")
                with col2:
                    if st.button(f"Drop rows with missing {col}", key=f"drop_{col}"):
                        # Store the current state before dropping
                        st.session_state.dropped_columns_history.append({
                            'column': col,
                            'timestamp': pd.Timestamp.now(),
                            'num_rows_before': len(df)
                        })
                        
                        # Drop rows with missing values in this column
                        df.dropna(subset=[col], inplace=True)
                        st.success(f"Dropped rows with missing values in {col}. Rows remaining: {len(df)}")
                        st.experimental_rerun()
            
            # Add undo functionality
            if st.session_state.dropped_columns_history:
                st.subheader("Undo History")
                st.write("You can undo the following operations:")
                
                for idx, action in enumerate(reversed(st.session_state.dropped_columns_history)):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"Dropped {action['column']} at {action['timestamp'].strftime('%H:%M:%S')}")
                    with col2:
                        if st.button("Undo", key=f"undo_{idx}"):
                            # Restore data to the state before this drop
                            df = st.session_state.original_data.copy()
                            # Reapply all drops except the one being undone
                            for hist_action in st.session_state.dropped_columns_history[:-idx-1]:
                                df.dropna(subset=[hist_action['column']], inplace=True)
                            # Update history
                            st.session_state.dropped_columns_history = st.session_state.dropped_columns_history[:-idx-1]
                            st.success(f"Undid dropping of {action['column']}")
                            st.experimental_rerun()
        else:
            st.success("No missing values found in the dataset!")

        # Handle duplicates
        st.subheader("🔍 Duplicates")
        
        # Initialize session state for tracking duplicate removals
        if 'duplicate_removal_history' not in st.session_state:
            st.session_state.duplicate_removal_history = []

        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            st.write(f"Found {duplicate_count} duplicate rows in the dataset.")
            
            # Show duplicate rows
            duplicate_df = df[df.duplicated(keep='first')]
            st.dataframe(duplicate_df, use_container_width=True)
            
            # Add options for handling duplicates
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                keep_option = st.selectbox(
                    "Keep which occurrence?",
                    ['first', 'last', 'none'],
                    help="'first' keeps first occurrence, 'last' keeps last occurrence, 'none' removes all duplicates"
                )
            with col2:
                subset_cols = st.multiselect(
                    "Consider only specific columns for duplicates?",
                    df.columns.tolist(),
                    help="Leave empty to check all columns"
                )
            with col3:
                if st.button("Remove Duplicates"):
                    # Store current state
                    st.session_state.duplicate_removal_history.append({
                        'timestamp': pd.Timestamp.now(),
                        'num_rows_before': len(df),
                        'keep': keep_option,
                        'subset': subset_cols if subset_cols else 'all columns'
                    })
                    
                    # Remove duplicates
                    df.drop_duplicates(
                        subset=subset_cols if subset_cols else None,
                        keep=keep_option,
                        inplace=True
                    )
                    st.success(f"Removed duplicates. Rows remaining: {len(df)}")
                    st.experimental_rerun()
            
            # Add undo functionality for duplicates
            if st.session_state.duplicate_removal_history:
                st.subheader("Undo Duplicate Removal History")
                st.write("You can undo the following operations:")
                
                for idx, action in enumerate(reversed(st.session_state.duplicate_removal_history)):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        subset_info = f" (subset: {action['subset']})" if isinstance(action['subset'], list) else ""
                        st.write(f"Removed duplicates keeping {action['keep']} occurrences{subset_info} at {action['timestamp'].strftime('%H:%M:%S')}")
                    with col2:
                        if st.button("Undo", key=f"undo_dup_{idx}"):
                            # Restore data to the state before this removal
                            df = st.session_state.original_data.copy()
                            # Reapply all operations except the one being undone
                            for hist_action in st.session_state.duplicate_removal_history[:-idx-1]:
                                df.drop_duplicates(
                                    subset=hist_action['subset'] if hist_action['subset'] != 'all columns' else None,
                                    keep=hist_action['keep'],
                                    inplace=True
                                )
                            # Update history
                            st.session_state.duplicate_removal_history = st.session_state.duplicate_removal_history[:-idx-1]
                            st.success(f"Undid duplicate removal operation")
                            st.experimental_rerun()
        else:
            st.success("No duplicate rows found in the dataset!")

        # Handle outliers
        st.subheader("📊 Outliers")
        
        # Initialize session state for tracking outlier removals
        if 'outlier_removal_history' not in st.session_state:
            st.session_state.outlier_removal_history = []

        # Get numeric columns for outlier detection
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if numeric_cols:
            st.write("Select a numeric column to check for outliers:")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_col = st.selectbox("Column", numeric_cols)
            with col2:
                method = st.selectbox(
                    "Detection Method",
                    ["IQR", "Z-Score"],
                    help="IQR: Uses 1.5 × InterQuartile Range, Z-Score: Uses standard deviations"
                )
            
            if selected_col:
                # Calculate outliers
                if method == "IQR":
                    Q1 = df[selected_col].quantile(0.25)
                    Q3 = df[selected_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
                    outlier_condition = (df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)
                else:  # Z-Score
                    z_scores = np.abs((df[selected_col] - df[selected_col].mean()) / df[selected_col].std())
                    outliers = df[z_scores > 3]
                    outlier_condition = z_scores > 3
                
                if len(outliers) > 0:
                    st.write(f"Found {len(outliers)} outliers in column '{selected_col}'")
                    
                    # Show outlier statistics
                    st.write("Outlier Statistics:")
                    stats_df = pd.DataFrame({
                        'Metric': ['Count', 'Minimum', 'Maximum', 'Mean', 'Median'],
                        'All Data': [
                            len(df),
                            df[selected_col].min(),
                            df[selected_col].max(),
                            df[selected_col].mean(),
                            df[selected_col].median()
                        ],
                        'Outliers': [
                            len(outliers),
                            outliers[selected_col].min(),
                            outliers[selected_col].max(),
                            outliers[selected_col].mean(),
                            outliers[selected_col].median()
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Show outlier rows
                    st.write("Outlier Rows:")
                    st.dataframe(outliers, use_container_width=True)
                    
                    # Add option to remove outliers
                    if st.button(f"Remove outliers from {selected_col}"):
                        # Store current state
                        st.session_state.outlier_removal_history.append({
                            'column': selected_col,
                            'method': method,
                            'timestamp': pd.Timestamp.now(),
                            'num_rows_before': len(df),
                            'num_outliers': len(outliers)
                        })
                        
                        # Remove outliers
                        df = df[~outlier_condition]
                        st.success(f"Removed {len(outliers)} outliers from {selected_col}. Rows remaining: {len(df)}")
                        st.experimental_rerun()
                    
                    # Add undo functionality for outliers
                    if st.session_state.outlier_removal_history:
                        st.subheader("Undo Outlier Removal History")
                        st.write("You can undo the following operations:")
                        
                        for idx, action in enumerate(reversed(st.session_state.outlier_removal_history)):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"Removed {action['num_outliers']} outliers from {action['column']} using {action['method']} at {action['timestamp'].strftime('%H:%M:%S')}")
                            with col2:
                                if st.button("Undo", key=f"undo_out_{idx}"):
                                    # Restore data to the state before this removal
                                    df = st.session_state.original_data.copy()
                                    # Reapply all operations except the one being undone
                                    for hist_action in st.session_state.outlier_removal_history[:-idx-1]:
                                        if hist_action['method'] == "IQR":
                                            Q1 = df[hist_action['column']].quantile(0.25)
                                            Q3 = df[hist_action['column']].quantile(0.75)
                                            IQR = Q3 - Q1
                                            lower_bound = Q1 - 1.5 * IQR
                                            upper_bound = Q3 + 1.5 * IQR
                                            df = df[~((df[hist_action['column']] < lower_bound) | (df[hist_action['column']] > upper_bound))]
                                        else:  # Z-Score
                                            z_scores = np.abs((df[hist_action['column']] - df[hist_action['column']].mean()) / df[hist_action['column']].std())
                                            df = df[~(z_scores > 3)]
                                    # Update history
                                    st.session_state.outlier_removal_history = st.session_state.outlier_removal_history[:-idx-1]
                                    st.success(f"Undid outlier removal from {action['column']}")
                                    st.experimental_rerun()
                else:
                    st.success(f"No outliers found in {selected_col} using {method} method!")
        else:
            st.warning("No numeric columns found for outlier detection.")

    with tab3:
        st.subheader("📈 Visualizations")
        
        # Get column types
        numeric_cols, categorical_cols, date_cols = get_column_types(df)
        
        # Visualization options
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Histogram", "Box Plot", "Bar Chart", "Scatter Plot", "Line Chart", "Pie Chart"]
        )
        
        if viz_type == "Histogram":
            if numeric_cols:
                col = st.selectbox("Select Column", numeric_cols)
                fig = px.histogram(df, x=col, title=f"Histogram of {col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns found for histogram.")
        
        elif viz_type == "Box Plot":
            if numeric_cols:
                col = st.selectbox("Select Column", numeric_cols)
                fig = px.box(df, y=col, title=f"Box Plot of {col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns found for box plot.")
        
        elif viz_type == "Bar Chart":
            if categorical_cols:
                col = st.selectbox("Select Column", categorical_cols)
                
                # Get top N categories if there are many unique values
                unique_count = df[col].nunique()
                if unique_count > 15:
                    top_n = st.slider("Select top N categories", 5, 20, 10)
                    top_categories = df[col].value_counts().nlargest(top_n).index
                    filtered_df = df[df[col].isin(top_categories)]
                    fig = px.bar(filtered_df[col].value_counts().reset_index(), 
                                 x='index', y=col, title=f"Top {top_n} values in {col}")
                else:
                    fig = px.bar(df[col].value_counts().reset_index(), 
                                 x='index', y=col, title=f"Bar Chart of {col}")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No categorical columns found for bar chart.")
        
        elif viz_type == "Scatter Plot":
            if len(numeric_cols) >= 2:
                col_x = st.selectbox("Select X-axis Column", numeric_cols)
                col_y = st.selectbox("Select Y-axis Column", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                
                # Optional color parameter
                color_col = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                if color_col == "None":
                    fig = px.scatter(df, x=col_x, y=col_y, title=f"Scatter Plot: {col_x} vs {col_y}")
                else:
                    fig = px.scatter(df, x=col_x, y=col_y, color=color_col, 
                                     title=f"Scatter Plot: {col_x} vs {col_y}, colored by {color_col}")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least two numeric columns for scatter plot.")
        
        elif viz_type == "Line Chart":
            if date_cols:
                date_col = st.selectbox("Select Date Column", date_cols)
                if numeric_cols:
                    y_col = st.selectbox("Select Y-axis Column", numeric_cols)
                    fig = px.line(df.sort_values(date_col), x=date_col, y=y_col, 
                                  title=f"Line Chart: {y_col} over {date_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric columns found for line chart.")
            elif numeric_cols:
                # Allow line chart with numeric x-axis if no date columns
                x_col = st.selectbox("Select X-axis Column", numeric_cols)
                y_options = [c for c in numeric_cols if c != x_col]
                if y_options:
                    y_col = st.selectbox("Select Y-axis Column", y_options)
                    fig = px.line(df.sort_values(x_col), x=x_col, y=y_col, 
                                  title=f"Line Chart: {y_col} vs {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least two numeric columns for line chart.")
            else:
                st.warning("No date or numeric columns found for line chart.")
        
        elif viz_type == "Pie Chart":
            if categorical_cols:
                col = st.selectbox("Select Column", categorical_cols)
                
                # Get top N categories if there are many unique values
                unique_count = df[col].nunique()
                if unique_count > 10:
                    top_n = st.slider("Select top N categories", 3, 15, 5)
                    value_counts = df[col].value_counts()
                    top_data = pd.DataFrame({
                        'Category': value_counts.nlargest(top_n).index,
                        'Count': value_counts.nlargest(top_n).values
                    })
                    # Add "Other" category
                    if len(value_counts) > top_n:
                        other_count = value_counts.nsmallest(len(value_counts) - top_n).sum()
                        other_df = pd.DataFrame({'Category': ['Other'], 'Count': [other_count]})
                        top_data = pd.concat([top_data, other_df])
                    
                    fig = px.pie(top_data, names='Category', values='Count', 
                                 title=f"Distribution of {col} (Top {top_n} categories)")
                else:
                    counts = df[col].value_counts().reset_index()
                    fig = px.pie(counts, names='index', values=col, title=f"Distribution of {col}")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No categorical columns found for pie chart.")
    
    with tab4:
        st.subheader("🔄 Correlation Analysis")
        
        if len(numeric_cols) > 1:
            # Correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Using plotly for heatmap
            fig = px.imshow(corr_matrix, 
                            text_auto=True, 
                            color_continuous_scale='RdBu_r',
                            title="Correlation Matrix")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top correlations
            st.subheader("Top Positive Correlations")
            # Get upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            # Stack the data and sort
            highest_corr = upper.stack().sort_values(ascending=False).head(10)
            st.dataframe(highest_corr)
            
            st.subheader("Top Negative Correlations")
            lowest_corr = upper.stack().sort_values().head(10)
            st.dataframe(lowest_corr)
        else:
            st.info("Need at least two numeric columns to calculate correlations.")
    
    with tab5:
        st.subheader("🔍 Data Explorer")
        
        # Filter options
        st.write("Filter your data")
        filter_container = st.container()
        
        with filter_container:
            col1, col2 = st.columns(2)
            
            with col1:
                # Column selection for filtering
                filter_col = st.selectbox("Select Column to Filter", df.columns)
            
            with col2:
                # Filter type depends on column data type
                if df[filter_col].dtype == 'object' or df[filter_col].nunique() < 10:
                    # Categorical filter
                    unique_values = df[filter_col].dropna().unique()
                    selected_values = st.multiselect("Select Values", unique_values, default=unique_values)
                    filtered_df = df[df[filter_col].isin(selected_values)]
                elif np.issubdtype(df[filter_col].dtype, np.number):
                    # Numeric filter
                    min_val = float(df[filter_col].min())
                    max_val = float(df[filter_col].max())
                    range_filter = st.slider(f"Range for {filter_col}", min_val, max_val, (min_val, max_val))
                    filtered_df = df[(df[filter_col] >= range_filter[0]) & (df[filter_col] <= range_filter[1])]
                else:
                    # Default case
                    filtered_df = df
        
        # Show filtered data
        st.subheader("Filtered Data")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Export options
        st.download_button(
            label="Export Filtered Data (CSV)",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name="filtered_data.csv",
            mime="text/csv"
        )

# Main execution flow
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:  # Excel file
            df = pd.read_excel(uploaded_file)
        analyze_data(df)
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Please check your file format and try again.")
else:
    # Display sample data option when no file is uploaded
    st.info("No file uploaded. You can try a sample dataset:")
    
    sample_dataset = st.selectbox(
        "Select a sample dataset",
        ["None", "Titanic Dataset", "Iris Flower Dataset", "Tips Dataset", "Gapminder Dataset"]
    )
    
    if sample_dataset == "Titanic Dataset":
        try:
            df = pd.read_csv("train.csv")
            st.write("Using Titanic dataset")
            analyze_data(df)
        except FileNotFoundError:
            st.error("Titanic dataset (train.csv) not found in the current directory.")
            st.info("Please ensure 'train.csv' is in the same directory as this script.")
    elif sample_dataset == "Iris Flower Dataset":
        df = px.data.iris()
        st.write("Using sample Iris dataset")
        analyze_data(df)
    elif sample_dataset == "Tips Dataset":
        df = px.data.tips()
        st.write("Using sample Tips dataset")
        analyze_data(df)
    elif sample_dataset == "Gapminder Dataset":
        df = px.data.gapminder()
        st.write("Using sample Gapminder dataset")
        analyze_data(df)