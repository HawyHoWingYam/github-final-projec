import pandas as pd
import plotly.express as px

csv_file_path = 'C:/Hawy/POLYU/PROJECT/5423/ADA-Group-11/data/train.csv'
# Load your dataset
df = pd.read_csv(csv_file_path)

# Define the column name for the sales price
sales_price_column = 'SalePrice'

# Filter out the sales price column and non-numeric columns from the columns to plot against
columns_to_plot = [col for col in df.columns if col != sales_price_column and col != 'Id' and pd.api.types.is_numeric_dtype(df[col])]

# Create an interactive 3D plot for the first three columns as an example
col_x = columns_to_plot[0]
col_y = columns_to_plot[1]
col_z = sales_price_column

# Create a 3D scatter plot using plotly
fig = px.scatter_3d(df, x=col_x, y=col_y, z=col_z)

# Show the plot
fig.show()
