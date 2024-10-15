import streamlit as st
import pandas as pd
import numpy as np
import time
from itertools import product
from collections import Counter, defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from PIL import Image
from pyecharts import options as opts
from pyecharts.charts import Calendar, Page
from pyecharts.globals import ThemeType
from pyecharts.commons.utils import JsCode
import streamlit.components.v1 as components
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, ColorBar
from bokeh.palettes import RdYlGn11
from bokeh.transform import linear_cmap

# Set page config
st.set_page_config(page_title="Shipment Optimization & Planning Dashboard", layout="wide")


logo = Image.open("perrigo-logo.png")
st.image(logo, width=120)

# Title
st.title("Shipment Optimization & Planning Dashboard")

# Load data and rate cards
@st.cache_data
def load_data():
    df = pd.read_excel('Complete Input.xlsx', sheet_name='Sheet1')
    df['SHIPPED_DATE'] = pd.to_datetime(df['SHIPPED_DATE'], dayfirst=True)
    rate_card_ambient = pd.read_excel('Complete Input.xlsx', sheet_name='AMBIENT')
    rate_card_ambcontrol = pd.read_excel('Complete Input.xlsx', sheet_name='AMBCONTROL')
    return df, rate_card_ambient, rate_card_ambcontrol

df, rate_card_ambient, rate_card_ambcontrol = load_data()

# Sidebar
st.sidebar.header("Parameters")

# Add slider for total shipment capacity
total_shipment_capacity = st.sidebar.slider("Total Shipment Capacity", 26, 52, 46)

# Group selection
group_method = st.sidebar.radio("Consolidation Level", ('Post Code Level', 'Customer Level'))
group_field = 'SHORT_POSTCODE' if group_method == 'Post Code Level' else 'NAME'

# Month selection
all_months = ['All', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
selected_month = st.sidebar.selectbox(
    "Select Month for Consolidation",
    options=all_months,
    index=1  # Default to 'All'
)

# Filter data based on selected month
if selected_month != 'All':
    month_num = all_months.index(selected_month)
    df = df[df['SHIPPED_DATE'].dt.month == month_num]

# Determine start_date and end_date based on selection
if selected_month == 'All':
    start_date = df['SHIPPED_DATE'].min()
    end_date = df['SHIPPED_DATE'].max()
else:
    # Convert month name to number
    month_to_num = {month: index for index, month in enumerate(all_months[1:], start=1)}
    selected_month_number = month_to_num[selected_month]
    
    # Find the year of the first occurrence of the selected month
    year = df[df['SHIPPED_DATE'].dt.month == selected_month_number]['SHIPPED_DATE'].dt.year.min()
    
    # Set start_date to the first day of the selected month
    start_date = pd.Timestamp(year=year, month=selected_month_number, day=1)
    
    # Set end_date to the last day of the selected month
    if selected_month_number == 12:
        end_date = pd.Timestamp(year=year, month=12, day=31)
    else:
        end_date = pd.Timestamp(year=year, month=selected_month_number + 1, day=1) - pd.Timedelta(days=1)


# Add checkbox and conditional dropdown for selecting post codes or customers
if group_method == 'Post Code Level':
    all_postcodes = st.sidebar.checkbox("All Post Codes", value=False)
    
    if not all_postcodes:
        postcode_counts = df['SHORT_POSTCODE'].value_counts()
        postcode_options = postcode_counts.index.tolist()
        selected_postcodes = st.sidebar.multiselect(
            "Select Post Codes",
            options=postcode_options,
            default=[postcode_options[0]],  # Start with an empty selection
            format_func=lambda x: f"{x} ({postcode_counts[x]})"
        )
else:  # Customer Level
    all_customers = st.sidebar.checkbox("All Customers", value=False)
    
    if not all_customers:
        customer_counts = df['NAME'].value_counts()
        customer_options = customer_counts.index.tolist()
        selected_customers = st.sidebar.multiselect(
            "Select Customers",
            options=customer_options,
            default=[customer_options[0]],  # Start with an empty selection
            format_func=lambda x: f"{x} ({customer_counts[x]})"
        )

# Filter the dataframe based on the selection
if group_method == 'Post Code Level' and not all_postcodes:
    if selected_postcodes:  # Only filter if some postcodes are selected
        df = df[df['SHORT_POSTCODE'].isin(selected_postcodes)]
    
elif group_method == 'Customer Level' and not all_customers:
    if selected_customers:  # Only filter if some customers are selected
        df = df[df['NAME'].isin(selected_customers)]
   

    
    
# Create tabs
tab1, tab2 = st.tabs(["Simulation", "Calculation"])



# Sidebar - Simulation Parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Parameters")
shipment_window_range = st.sidebar.slider("Shipment Window", 0, 7, (2, 4))
step_size = st.sidebar.selectbox("Utilization Threshold Step Size", [5, 10, 15, 20], index=0)
utilization_threshold_range = st.sidebar.slider("Utilization Threshold", 25, 95, (50, 80))


# Sidebar - Calculation Parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Calculation Parameters")
calc_shipment_window = st.sidebar.number_input("Shipment Window", min_value=0, max_value=7, value=4)
calc_high_priority_limit = st.sidebar.number_input("High Priority Limit", min_value=0, max_value=7, value=2)
calc_utilization_threshold = st.sidebar.number_input("Utilization Threshold", min_value=25, max_value=95, value=70, step=5)


# Helper functions
def calculate_priority(shipped_date, current_date, shipment_window):
    days_left = (shipped_date - current_date).days
    if 0 <= days_left <= shipment_window:
        return days_left
    return np.nan

def get_shipment_cost(prod_type, short_postcode, total_pallets):
    if prod_type == 'AMBIENT':
        rate_card = rate_card_ambient
    elif prod_type == 'AMBCONTROL':
        rate_card = rate_card_ambcontrol
    else:
        return np.nan

    row = rate_card[rate_card['SHORT_POSTCODE'] == short_postcode]
    
    if row.empty:
        return np.nan

    cost_per_pallet = row.get(total_pallets, np.nan).values[0]

    if pd.isna(cost_per_pallet):
        return np.nan

    shipment_cost = round(cost_per_pallet * total_pallets, 1)
    return shipment_cost

def get_baseline_cost(prod_type, short_postcode, pallets):
    total_cost = 0
    for pallet in pallets:
        cost = get_shipment_cost(prod_type, short_postcode, pallet)
        if pd.isna(cost):
            return np.nan
        total_cost += cost
    return round(total_cost, 1)

def best_fit_decreasing(items, capacity):
    items = sorted(items, key=lambda x: x['Total Pallets'], reverse=True)
    shipments = []

    for item in items:
        best_shipment = None
        min_space = capacity + 1

        for shipment in shipments:
            current_load = sum(order['Total Pallets'] for order in shipment)
            new_load = current_load + item['Total Pallets']
            
            if new_load <= capacity:
                space_left = capacity - current_load
            else:
                continue  # Skip this shipment if adding the item would exceed capacity
            
            if item['Total Pallets'] <= space_left < min_space:
                best_shipment = shipment
                min_space = space_left

        if best_shipment:
            best_shipment.append(item)
        else:
            shipments.append([item])

    return shipments

def process_shipment(shipment, consolidated_shipments, allocation_matrix, working_df, current_date, capacity):
    total_pallets = sum(order['Total Pallets'] for order in shipment)
    utilization = (total_pallets / capacity) * 100

    prod_type = shipment[0]['PROD TYPE']
    short_postcode = shipment[0]['SHORT_POSTCODE']
    shipment_cost = get_shipment_cost(prod_type, short_postcode, total_pallets)

    pallets = [order['Total Pallets'] for order in shipment]
    baseline_cost = get_baseline_cost(prod_type, short_postcode, pallets)

    shipment_info = {
        'Date': current_date,
        'Orders': [order['ORDER_ID'] for order in shipment],
        'Total Pallets': total_pallets,
        'Capacity': capacity,
        'Utilization %': round(utilization, 1),
        'Order Count': len(shipment),
        'Pallets': pallets,
        'PROD TYPE': prod_type,
        'GROUP': shipment[0]['GROUP'],
        'Shipment Cost': shipment_cost,
        'Baseline Cost': baseline_cost,
        'SHORT_POSTCODE': short_postcode,
        'Load Type': 'Full' if total_pallets > 26 else 'Partial'
    }

    if group_method == 'NAME':
        shipment_info['NAME'] = shipment[0]['NAME']

    consolidated_shipments.append(shipment_info)
    
    for order in shipment:
        allocation_matrix.loc[order['ORDER_ID'], current_date] = 1
        working_df.drop(working_df[working_df['ORDER_ID'] == order['ORDER_ID']].index, inplace=True)

def consolidate_shipments(df, high_priority_limit, utilization_threshold, shipment_window, date_range, progress_callback, capacity):
    consolidated_shipments = []
    allocation_matrix = pd.DataFrame(0, index=df['ORDER_ID'], columns=date_range)
    
    working_df = df.copy()
    
    for current_date in date_range:
        working_df.loc[:, 'Priority'] = working_df['SHIPPED_DATE'].apply(lambda x: calculate_priority(x, current_date, shipment_window))

        if (working_df['Priority'] == 0).any():
            eligible_orders = working_df[working_df['Priority'].notnull()].sort_values('Priority')
            high_priority_orders = eligible_orders[eligible_orders['Priority'] <= high_priority_limit].to_dict('records')
            low_priority_orders = eligible_orders[eligible_orders['Priority'] > high_priority_limit].to_dict('records')
            
            if high_priority_orders or low_priority_orders:
                # Process high priority orders first
                high_priority_shipments = best_fit_decreasing(high_priority_orders, capacity)
                
                # Try to fill high priority shipments with low priority orders
                for shipment in high_priority_shipments:
                    current_load = sum(order['Total Pallets'] for order in shipment)
                    space_left = capacity - current_load  # Use the variable capacity
                    
                    if space_left > 0:
                        low_priority_orders.sort(key=lambda x: x['Total Pallets'], reverse=True)
                        for low_priority_order in low_priority_orders[:]:
                            if low_priority_order['Total Pallets'] <= space_left:
                                shipment.append(low_priority_order)
                                space_left -= low_priority_order['Total Pallets']
                                low_priority_orders.remove(low_priority_order)
                            if space_left == 0:
                                break
                
                # Process remaining low priority orders
                low_priority_shipments = best_fit_decreasing(low_priority_orders, capacity)
                
                # Process all shipments
                all_shipments = high_priority_shipments + low_priority_shipments
                for shipment in all_shipments:
                    total_pallets = sum(order['Total Pallets'] for order in shipment)
                    utilization = (total_pallets / capacity) * 100
                    
                    # Always process shipments with high priority orders, apply threshold only to pure low priority shipments
                    if any(order['Priority'] <= high_priority_limit for order in shipment) or utilization >= utilization_threshold:
                        process_shipment(shipment, consolidated_shipments, allocation_matrix, working_df, current_date, capacity)
        
        progress_callback()
    
    return consolidated_shipments, allocation_matrix

def calculate_metrics(all_consolidated_shipments, df):
    total_orders = sum(len(shipment['Orders']) for shipment in all_consolidated_shipments)
    total_shipments = len(all_consolidated_shipments)
    total_pallets = sum(shipment['Total Pallets'] for shipment in all_consolidated_shipments)
    total_utilization = sum(shipment['Utilization %'] for shipment in all_consolidated_shipments)
    average_utilization = total_utilization / total_shipments if total_shipments > 0 else 0
    total_shipment_cost = sum(shipment['Shipment Cost'] for shipment in all_consolidated_shipments if not pd.isna(shipment['Shipment Cost']))
    total_baseline_cost = sum(shipment['Baseline Cost'] for shipment in all_consolidated_shipments if not pd.isna(shipment['Baseline Cost']))
    cost_savings = total_baseline_cost - total_shipment_cost
    percent_savings = (cost_savings / total_baseline_cost) * 100 if total_baseline_cost > 0 else 0

    # Calculate CO2 Emission
    total_distance = 0
    for shipment in all_consolidated_shipments:
        order_ids = shipment['Orders']
        avg_distance = df[df['ORDER_ID'].isin(order_ids)]['Distance'].mean()
        total_distance += avg_distance
    co2_emission = total_distance * 2  # Multiply by 2 


    metrics = {
        'Total Orders': total_orders,
        'Total Shipments': total_shipments,
        'Total Pallets': total_pallets,
        'Average Utilization': average_utilization,
        'Total Shipment Cost': total_shipment_cost,
        'Total Baseline Cost': total_baseline_cost,
        'Cost Savings': cost_savings,
        'Percent Savings': percent_savings,
        'CO2 Emission': co2_emission
    }

    return metrics

def create_utilization_chart(all_consolidated_shipments):
    utilization_bins = {f"{i}-{i+5}%": 0 for i in range(0, 100, 5)}
    for shipment in all_consolidated_shipments:
        utilization = shipment['Utilization %']
        bin_index = min(int(utilization // 5) * 5, 95)  # Cap at 95-100% bin
        bin_key = f"{bin_index}-{bin_index+5}%"
        utilization_bins[bin_key] += 1

    total_shipments = len(all_consolidated_shipments)
    utilization_distribution = {bin: (count / total_shipments) * 100 for bin, count in utilization_bins.items()}

    fig = go.Figure(data=[go.Bar(x=list(utilization_distribution.keys()), y=list(utilization_distribution.values()))])
    fig.update_layout(title='Utilization Distribution', xaxis_title='Utilization Range', yaxis_title='Percentage of Shipments')
    return fig

def create_pallet_distribution_chart(all_consolidated_shipments, total_shipment_capacity):
    # Create bins with 5-pallet intervals
    bin_size = 5
    num_bins = (total_shipment_capacity + bin_size - 1) // bin_size  # Round up to nearest bin
    pallet_bins = {f"{i*bin_size+1}-{min((i+1)*bin_size, total_shipment_capacity)}": 0 for i in range(num_bins)}

    for shipment in all_consolidated_shipments:
        total_pallets = shipment['Total Pallets']
        for bin_range, count in pallet_bins.items():
            low, high = map(int, bin_range.split('-'))
            if low <= total_pallets <= high:
                pallet_bins[bin_range] += 1
                break

    total_shipments = len(all_consolidated_shipments)
    pallet_distribution = {bin: (count / total_shipments) * 100 for bin, count in pallet_bins.items()}

    # Sort the bins to ensure they're in the correct order
    sorted_bins = sorted(pallet_distribution.items(), key=lambda x: int(x[0].split('-')[0]))

    fig = go.Figure(data=[go.Bar(x=[bin for bin, _ in sorted_bins], y=[value for _, value in sorted_bins])])
    fig.update_layout(
        title='Pallet Distribution',
        xaxis_title='Pallet Range',
        yaxis_title='Percentage of Shipments',
        xaxis=dict(tickangle=-45)
    )
    return fig


def create_consolidated_shipments_calendar(consolidated_df):
    # Group by Date and calculate both Shipments Count and Total Orders
    df_consolidated = consolidated_df.groupby('Date').agg({
        'Orders': ['count', lambda x: sum(len(orders) for orders in x)]  # Count rows for shipments, sum order lengths for total orders
    }).reset_index()
    df_consolidated.columns = ['Date', 'Shipments Count', 'Orders Count']
    
    calendar_data_consolidated = df_consolidated[['Date', 'Shipments Count', 'Orders Count']].values.tolist()

    calendar = (
        Calendar(init_opts=opts.InitOpts(width="1300px", height="220px", theme=ThemeType.ROMANTIC))
        .add(
            series_name="",
            yaxis_data=calendar_data_consolidated,
            calendar_opts=opts.CalendarOpts(
                pos_top="50",
                pos_left="40",
                pos_right="30",
                range_=str(calendar_data_consolidated[0][0].year),
                yearlabel_opts=opts.CalendarYearLabelOpts(is_show=False),
                daylabel_opts=opts.CalendarDayLabelOpts(name_map=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']),
                monthlabel_opts=opts.CalendarMonthLabelOpts(name_map="en"),
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Calendar Heatmap for Orders and Shipments Consolidated"),
            visualmap_opts=opts.VisualMapOpts(
                max_=max(item[2] for item in calendar_data_consolidated),  # Changed from item[1] to item[2]
                min_=min(item[2] for item in calendar_data_consolidated),  # Changed from item[1] to item[2]
                orient="horizontal",
                is_piecewise=False,
                pos_bottom="20",
                pos_left="center",
                range_color=["#E8F5E9", "#1B5E20"],
                is_show=False,
            ),
            tooltip_opts=opts.TooltipOpts(
                formatter=JsCode(
                    """
                    function (p) {
                        var date = new Date(p.data[0]);
                        var day = date.getDate().toString().padStart(2, '0');
                        var month = (date.getMonth() + 1).toString().padStart(2, '0');
                        var year = date.getFullYear();
                        return 'Date: ' + day + '/' + month + '/' + year + 
                               '<br/>Orders: ' + p.data[2] +
                               '<br/>Shipments: ' + p.data[1];
                    }
                    """
                )
            )
        )
    )
    return calendar

def create_original_orders_calendar(original_df):
    df_original = original_df.groupby('SHIPPED_DATE').size().reset_index(name='Orders Shipped')
    calendar_data_original = df_original[['SHIPPED_DATE', 'Orders Shipped']].values.tolist()

    calendar = (
        Calendar(init_opts=opts.InitOpts(width="1300px", height="220px", theme=ThemeType.ROMANTIC))
        .add(
            series_name="",
            yaxis_data=calendar_data_original,
            calendar_opts=opts.CalendarOpts(
                pos_top="50",
                pos_left="40",
                pos_right="30",
                range_=str(calendar_data_original[0][0].year),
                yearlabel_opts=opts.CalendarYearLabelOpts(is_show=False),
                daylabel_opts=opts.CalendarDayLabelOpts(name_map=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']),
                monthlabel_opts=opts.CalendarMonthLabelOpts(name_map="en"),
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Calendar Heatmap for Orders Shipped Before Consolidation"),
            visualmap_opts=opts.VisualMapOpts(
                max_=max(item[1] for item in calendar_data_original),
                min_=min(item[1] for item in calendar_data_original),
                orient="horizontal",
                is_piecewise=False,
                pos_bottom="20",
                pos_left="center",
                range_color=["#E8F5E9", "#1B5E20"],
                is_show=False,
            ),
            tooltip_opts=opts.TooltipOpts(
                formatter=JsCode(
                    """
                    function (p) {
                        var date = new Date(p.data[0]);
                        var day = date.getDate().toString().padStart(2, '0');
                        var month = (date.getMonth() + 1).toString().padStart(2, '0');
                        var year = date.getFullYear();
                        return 'Date: ' + day + '/' + month + '/' + year + '<br/>Orders: ' + p.data[1];
                    }
                    """
                )
            )
        )
    )
    return calendar

def create_heatmap_charts(consolidated_df, original_df):
    chart_consolidated = create_consolidated_shipments_calendar(consolidated_df)
    chart_original = create_original_orders_calendar(original_df)

    page = Page(layout=Page.SimplePageLayout)
    page.add(chart_consolidated, chart_original)

    return page


def analyze_consolidation_distribution(all_consolidated_shipments, df):
    distribution = {}
    for shipment in all_consolidated_shipments:
        consolidation_date = shipment['Date']
        for order_id in shipment['Orders']:
            shipped_date = df.loc[df['ORDER_ID'] == order_id, 'SHIPPED_DATE'].iloc[0]
            days_difference = (shipped_date - consolidation_date).days
            if days_difference not in distribution:
                distribution[days_difference] = 0
            distribution[days_difference] += 1
    
    total_orders = sum(distribution.values())
    distribution_percentage = {k: round((v / total_orders) * 100, 1) for k, v in distribution.items()}
    return distribution, distribution_percentage


def custom_progress_bar():
    progress_container = st.empty()
    
    def render_progress(percent):
        progress_html = f"""
        <style>
            .overall-container {{
                width: 100%;
                position: relative;
                padding-top: 30px; /* Reduced space for the truck */
            }}
            .progress-container {{
                width: 100%;
                height: 8px;
                background-color: #bbddff;
                border-radius: 10px;
                position: relative;
                overflow: hidden;
            }}
            .progress-bar {{
                width: {percent}%;
                height: 100%;
                background-color: #0053a4;
                border-radius: 10px;
                transition: width 0.5s ease-in-out;
            }}
            .truck-icon {{
                position: absolute;
                top: 0;
                left: calc({percent}% - 15px);
                transition: left 0.5s ease-in-out;
            }}
        </style>
        <div class="overall-container">
            <div class="truck-icon">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 512" width="30" height="30">
                    <path fill="#0053a4" d="M624 352h-16V243.9c0-12.7-5.1-24.9-14.1-33.9L494 110.1c-9-9-21.2-14.1-33.9-14.1H416V48c0-26.5-21.5-48-48-48H48C21.5 0 0 21.5 0 48v320c0 26.5 21.5 48 48 48h16c0 53 43 96 96 96s96-43 96-96h128c0 53 43 96 96 96s96-43 96-96h48c8.8 0 16-7.2 16-16v-32c0-8.8-7.2-16-16-16zm-464 96c-26.5 0-48-21.5-48-48s21.5-48 48-48 48 21.5 48 48-21.5 48-48 48zm208-96H242.7c-16.6-28.6-47.2-48-82.7-48s-66.1 19.4-82.7 48H48V48h352v304zm96 96c-26.5 0-48-21.5-48-48s21.5-48 48-48 48 21.5 48 48-21.5 48-48 48zm80-96h-16.7c-16.6-28.6-47.2-48-82.7-48-29.2 0-55.1 14.2-71.3 36-3.1-1.9-6.4-3.5-9.9-4.7V144h80.9c4.7 0 9.2 1.9 12.5 5.2l100.2 100.2c2.1 2.1 3.3 5 3.3 8v95.8z"/>
                </svg>
            </div>
            <div class="progress-container">
                <div class="progress-bar"></div>
            </div>
        </div>
        """
        progress_container.markdown(progress_html, unsafe_allow_html=True)
    
    return render_progress


def create_metric_box(label, value, color_start="#1f77b4", color_end="#0053a4"):
    html = f"""
    <div style="
        background: linear-gradient(135deg, {color_start} 0%, {color_end} 100%);
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <h4 style="color: white; margin: 0; font-size: 22px; font-weight: 500;">{label}</h4>
        <p style="color: white; font-size: 22px; font-weight: 600; margin: 5px 0 0 0;">{value}</p>
    </div>
    """
    return st.markdown(html, unsafe_allow_html=True)


st.markdown("""
<style>
/* Style for regular buttons */
div.stButton > button:first-child {
    background-color: #c06238;
    color: white;
}
div.stButton > button:hover {
    background-color: #0053a4;
    color: white;
}

/* Style for download buttons */
div.stDownloadButton > button:first-child {
    background-color: #c06238;
    color: white;
}
div.stDownloadButton > button:hover {
    background-color: #0053a4;
    color: white;
}
</style>
""", unsafe_allow_html=True)

    
# Simulation tab
with tab1:
    st.header("Simulation Results")
    
    with st.expander("Iteration Breakdown", expanded=False):

        # Calculate the values dynamically
        total_days = (end_date - start_date).days + 1
        grouped = df.groupby(['PROD TYPE', group_field])
        total_groups = len(grouped)
        utilization_thresholds = range(utilization_threshold_range[0], utilization_threshold_range[1] + 1, step_size)
        shipment_windows = range(shipment_window_range[0], shipment_window_range[1] + 1)
    
        # Display the breakdown
        st.write(f"Total days to check consolidation: {total_days} ({start_date.date()} to {end_date.date()})")
        st.write(f"Total groups (PROD TYPE, {group_field}): {total_groups}")
        st.write(f"Utilization thresholds: {', '.join(map(str, utilization_thresholds))}")
        st.write(f"Shipment windows: {', '.join(map(str, shipment_windows))}")
        
        st.write("High priority limits:")
        for window in shipment_windows:
            st.write(f"  For shipment window {window}: {', '.join(map(str, range(window + 1)))}")
        
        # Calculate total iterations
        total_iterations = sum(len(range(shipment_window + 1)) * len(utilization_thresholds) * total_groups * total_days for shipment_window in shipment_windows)
        total_calculations = sum(len(range(shipment_window + 1)) * len(utilization_thresholds) for shipment_window in shipment_windows)
        st.write(f"\nTotal iterations: {total_iterations:,} (Total Calculations: {total_calculations:,})")    
    

    if st.button("Run Simulation"):
        start_time = time.time()
        
        with st.spinner("Running simulation..."):
            # Prepare data for simulation
            df['GROUP'] = df[group_field]
            grouped = df.groupby(['PROD TYPE', 'GROUP'])
            date_range = pd.date_range(start=start_date, end=end_date)
            
            # Generate all combinations of parameters
            shipment_windows = range(shipment_window_range[0], shipment_window_range[1] + 1)
            utilization_thresholds = range(utilization_threshold_range[0], utilization_threshold_range[1] + 1, step_size)
            
            total_groups = len(grouped)
            total_days = len(date_range)
            total_iterations = sum(len(range(shipment_window + 1)) * len(utilization_thresholds) * total_groups * total_days for shipment_window in shipment_windows)
    
            # Initialize variables to store best results
            best_metrics = None
            best_consolidated_shipments = None
            best_params = None
            
            # Create a progress bar
            progress_bar = custom_progress_bar()
            iteration_counter = 0
            
            all_results = []
            
            # Run simulation for each combination
            for shipment_window in shipment_windows:
                for utilization_threshold in utilization_thresholds:
                    for high_priority_limit in range(shipment_window + 1):
                        all_consolidated_shipments = []
                    
                        for (prod_type, group), group_df in grouped:
                            consolidated_shipments, _ = consolidate_shipments(
                                group_df, high_priority_limit, utilization_threshold, 
                                shipment_window, date_range, lambda: None, total_shipment_capacity
                            )
                            all_consolidated_shipments.extend(consolidated_shipments)
                            
                            iteration_counter += total_days
                            progress_percentage = int((iteration_counter / total_iterations) * 100)
                            progress_bar(progress_percentage)
                    
                        # Calculate metrics for this combination
                        metrics = calculate_metrics(all_consolidated_shipments, df)
                        
                        # Analyze consolidation distribution
                        distribution, distribution_percentage = analyze_consolidation_distribution(all_consolidated_shipments, df)
                        
                        result = {
                            'Shipment Window': shipment_window,
                            'High Priority Limit': high_priority_limit,
                            'Utilization Threshold': utilization_threshold,
                            'Total Orders': metrics['Total Orders'],
                            'Total Shipments': metrics['Total Shipments'],
                            'Total Shipment Cost': metrics['Total Shipment Cost'],
                            'Total Baseline Cost': metrics['Total Baseline Cost'],
                            'Cost Savings': metrics['Cost Savings'],
                            'Percent Savings': round(metrics['Percent Savings'], 1),
                            'Average Utilization': round(metrics['Average Utilization'], 1),
                            'CO2 Emission': round(metrics['CO2 Emission'], 1)
                        }
                        
                        # Add columns for days relative to shipped date
                        for i in range(shipment_window + 1):
                            column_name = f'orders%_shipping_{i}day_early'
                            result[column_name] = distribution_percentage.get(i, 0)
                        
                        all_results.append(result)
                    
                        # Update best results if current combination is better
                        if best_metrics is None or metrics['Cost Savings'] > best_metrics['Cost Savings']:
                            best_metrics = metrics
                            best_consolidated_shipments = all_consolidated_shipments
                            best_params = (shipment_window, high_priority_limit, utilization_threshold)
                    
                    
            
            end_time = time.time()
            time_taken = end_time - start_time
            time_taken_minutes = int(time_taken // 60)
            time_taken_seconds = int(time_taken % 60)
            st.write(f"Time taken: {time_taken_minutes} minutes {time_taken_seconds} seconds")            
                
            # Display best results
            st.subheader("Best Simulation Results")
            st.write(f"Best Parameters: Shipment Window = {best_params[0]}, High Priority Limit = {best_params[1]}, Utilization Threshold = {best_params[2]}%")
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                create_metric_box("Total Orders", f"{best_metrics['Total Orders']:,}")
            with col2:
                create_metric_box("Total Shipments", f"{best_metrics['Total Shipments']:,}")
            with col3:
                create_metric_box("Cost Savings", f"£{best_metrics['Cost Savings']:,.1f}")
            with col4:
                create_metric_box("Percentage Savings", f"{best_metrics['Percent Savings']:.1f}%")
            with col5:
                create_metric_box("Average Utilization", f"{best_metrics['Average Utilization']:.1f}%")
            with col6:
                create_metric_box("CO2 Emission", f"{best_metrics['CO2 Emission']:,.1f}")
                
    
            # Display charts for best results
            st.subheader("Utilization Distribution (Best Results)")
            st.plotly_chart(create_utilization_chart(best_consolidated_shipments), use_container_width=True)
            
            st.subheader("Pallet Distribution (Best Results)")
            fig_pallet_distribution = create_pallet_distribution_chart(all_consolidated_shipments, total_shipment_capacity)
            st.plotly_chart(fig_pallet_distribution, use_container_width=True)

            # Create a dataframe with all simulation results
            results_df = pd.DataFrame(all_results)
            
            
            # Preprocess the data to keep only the row with max Cost Savings for each Shipment Window and Utilization Threshold
            optimal_results = results_df.loc[results_df.groupby(['Shipment Window', 'Utilization Threshold'])['Cost Savings'].idxmax()]
            
            # Create ColumnDataSource
            source = ColumnDataSource(optimal_results)
            
            # Create color mapper for Cost Savings (green for high savings, red for low savings)
            color_mapper = linear_cmap(field_name='Cost Savings', palette=RdYlGn11[::-1], 
                                       low=optimal_results['Cost Savings'].min(), high=optimal_results['Cost Savings'].max())
            
            # Create the figure
            p = figure(title="Optimal Simulation Results: Shipment Window vs Utilization Threshold",
                       x_axis_label='Utilization Threshold (%)',
                       y_axis_label='Shipment Window',
                       width=800, height=600)
            
            # Add the scatter plot
            scatter = p.scatter('Utilization Threshold', 'Shipment Window', source=source,
                                size=10, color=color_mapper, alpha=0.8)
            
            # Add color bar
            color_bar = ColorBar(color_mapper=color_mapper['transform'], width=8, location=(0,0),
                                 title="Cost Savings (£)")
            p.add_layout(color_bar, 'right')
            
            # Add hover tool
            hover = HoverTool(tooltips=[
                ('Shipment Window', '@{Shipment Window}'),
                ('Utilization Threshold', '@{Utilization Threshold}%'),
                ('High Priority Limit', '@{High Priority Limit}'),
                ('Total Shipment Cost', '£@{Total Shipment Cost}{,.1f}'),
                ('Cost Savings', '£@{Cost Savings}{,.1f}'),
                ('Percent Savings', '@{Percent Savings}{,.1f}%'),
                ('Total Shipments', '@{Total Shipments}'),
                ('Average Utilization', '@{Average Utilization}{,.1f}%'),
                ('CO2 Emission', '@{CO2 Emission}{,.1f}'),
                ('Orders Shipping 0 Day Early', '@{orders%_shipping_0day_early}{,.1f}%'),
            ])
            p.add_tools(hover)
            
            # Add labels for Percent Savings
            labels = p.text('Utilization Threshold', 'Shipment Window', text='Percent Savings',
                            x_offset=5, y_offset=5, source=source, text_font_size='8pt')
            
            # Display the plot in Streamlit
            st.bokeh_chart(p, use_container_width=True)


            
            # Display the Shipment Window Comparison chart
            st.subheader("Shipment Window Comparison")
            
            # Select the best rows for each shipment window
            best_results = results_df.loc[results_df.groupby('Shipment Window')['Percent Savings'].idxmax()]
            
            # Sort by Shipment Window
            best_results = best_results.sort_values('Shipment Window')
            
            # Calculate the figure width based on the number of shipment windows
            bar_width = 100  # Width of each bar in pixels
            gap_width = 50  # Width of gap between bar groups in pixels
            margin_width = 200  # Extra width for margins and labels
            num_windows = len(best_results)
            figure_width = (bar_width + gap_width) * num_windows + margin_width
            
            # Create the subplot figure
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add the stacked bar chart
            fig.add_trace(
                go.Bar(
                    x=best_results['Shipment Window'],
                    y=best_results['Total Shipment Cost'],
                    name='Total Shipment Cost',
                    marker_color='#1f77b4',
                    width=0.8  # This is relative width, 1 would mean no gap between bars
                )
            )
            
            fig.add_trace(
                go.Bar(
                    x=best_results['Shipment Window'],
                    y=best_results['Cost Savings'],
                    name='Cost Savings',
                    marker_color='#a9d6a9',
                    width=0.8
                )
            )
            
            # Add the line chart for Total Shipments on secondary y-axis
            fig.add_trace(
                go.Scatter(
                    x=best_results['Shipment Window'],
                    y=best_results['Total Shipments'],
                    name='Total Shipments',
                    mode='lines+markers',
                    line=dict(color='#ff7f0e', width=2),
                    marker=dict(size=8),
                    hovertemplate='<b>Shipment Window</b>: %{x}<br>' +
                                  '<b>Total Shipments</b>: %{y}<br>' +
                                  '<b>Average Utilization</b>: %{text:.1f}%<extra></extra>',
                    text=best_results['Average Utilization'],
                ),
                secondary_y=True
            )
            
            # Add text annotations for Percent Savings
            for i, row in best_results.iterrows():
                fig.add_annotation(
                    x=row['Shipment Window'],
                    y=row['Total Shipment Cost'] + row['Cost Savings'],
                    text=f"{row['Percent Savings']:.1f}%",
                    showarrow=False,
                    yanchor='bottom',
                    yshift=5,
                    font=dict(size=10)
                )
            
            # Update the layout
            fig.update_layout(
                barmode='stack',
                height=600,
                width=figure_width,
                margin=dict(l=50, r=50, t=40, b=20),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                bargap=0.2  # Gap between bars in a group
            )
            
            fig.update_xaxes(title_text='Shipment Window', tickmode='linear', dtick=1)
            fig.update_yaxes(title_text='Cost (£)', secondary_y=False)
            fig.update_yaxes(title_text='Total Shipments', secondary_y=True)
            
            # Show the chart
            st.plotly_chart(fig, use_container_width=False)
            
            
            # Display table with all results
            results_df = results_df.sort_values(by=['Percent Savings', 'Utilization Threshold'], ascending=[False, False]).reset_index(drop=True).set_index('Shipment Window')
            st.subheader("All Simulation Results")
            
            # Custom CSS to set the height of the dataframe
            custom_css = """
                <style>
                    .stDataFrame {
                        max-height: 250px;
                        overflow-y: auto;
                    }
                </style>
            """
            st.markdown(custom_css, unsafe_allow_html=True)
            
            # Display the dataframe
            st.dataframe(results_df)
            
            # Download results as CSV
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Simulation Results CSV",
                data=csv,
                file_name="simulation_results.csv",
                mime="text/csv",
            )
            

# Calculation tab
with tab2:
    st.header("Calculation Results")
    
    with st.expander("Iteration Breakdown", expanded=False):

        # Calculate the values dynamically
        total_days = (end_date - start_date).days + 1
        grouped = df.groupby(['PROD TYPE', group_field])
        total_groups = len(grouped)
        total_iterations = total_days * total_groups
    
        # Display the breakdown
        st.write(f"Total days to check consolidation: {total_days} ({start_date.date()} to {end_date.date()})")
        st.write(f"Total groups (PROD TYPE, {group_field}): {total_groups}")
        st.write(f"Total iterations: {total_iterations:,}")
    
    
# Add a section for data exploration
st.header("Data Exploration")
if st.checkbox("Show raw data"):
    st.subheader("Raw data")
    st.write(df)


# Add a histogram of pallets per order
st.subheader("Distribution of Pallets per Order")
fig_pallets = px.histogram(df, x="Total Pallets", nbins=50, title="Distribution of Pallets per Order")
st.plotly_chart(fig_pallets, use_container_width=True)

st.sidebar.info('This dashboard provides insights into shipment consolidation for Perrigo. Use tabs to switch between simulation and calculation modes.')
