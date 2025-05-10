import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import plotly.figure_factory as ff

# ------------------- PAGE SETUP -------------------
st.set_page_config(page_title="Heart Health Dashboard", page_icon='Heart-Disease.png', layout="wide")

# ------------------- GLOBAL STYLE -------------------
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #0f172a;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 1rem;
        border-radius: 16px;
        color: #60a5fa;
        font-weight: 600;
        box-shadow: 0 0 8px rgba(96,165,250,0.4);
    }
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-radius: 16px;
        padding: 1rem !important;
        box-shadow: 0 0 12px rgba(96,165,250,0.4);
    }
    div[data-baseweb="select"] {
        background-color: #1e293b !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
        box-shadow: 0 0 8px rgba(96,165,250,0.4);
    }
    div[data-baseweb="select"] * {
        color: white !important;
        background-color: #1e293b !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- LOAD DATA -------------------
df = pd.read_csv("dashboard\\train_balanced.csv")
df["HadHeartAttack"] = df["HadHeartAttack"].map({1: "yes", 0: "no"})
df = df.dropna()
df2 = pd.read_csv("dashboard\\transformed_train_data.csv")



state_coords = {
    'Alabama': [32.806671, -86.791130], 'Alaska': [61.370716, -152.404419],
    'Arizona': [33.729759, -111.431221], 'Arkansas': [34.969704, -92.373123],
    'California': [36.116203, -119.681564], 'Colorado': [39.059811, -105.311104],
    'Connecticut': [41.597782, -72.755371], 'Delaware': [39.318523, -75.507141],
    'District of Columbia': [38.897438, -77.026817], 'Florida': [27.766279, -81.686783],
    'Georgia': [33.040619, -83.643074], 'Hawaii': [21.094318, -157.498337],
    'Idaho': [44.240459, -114.478828], 'Illinois': [40.349457, -88.986137],
    'Indiana': [39.849426, -86.258278], 'Iowa': [42.011539, -93.210526],
    'Kansas': [38.526600, -96.726486], 'Kentucky': [37.668140, -84.670067],
    'Louisiana': [31.169546, -91.867805], 'Maine': [44.693947, -69.381927],
    'Maryland': [39.063946, -76.802101], 'Massachusetts': [42.230171, -71.530106],
    'Michigan': [43.326618, -84.536095], 'Minnesota': [45.694454, -93.900192],
    'Mississippi': [32.741646, -89.678696], 'Missouri': [38.456085, -92.288368],
    'Montana': [46.921925, -110.454353], 'Nebraska': [41.125370, -98.268082],
    'Nevada': [38.313515, -117.055374], 'New Hampshire': [43.452492, -71.563896],
    'New Jersey': [40.298904, -74.521011], 'New Mexico': [34.840515, -106.248482],
    'New York': [42.165726, -74.948051], 'North Carolina': [35.630066, -79.806419],
    'North Dakota': [47.528912, -99.784012], 'Ohio': [40.388783, -82.764915],
    'Oklahoma': [35.565342, -96.928917], 'Oregon': [44.572021, -122.070938],
    'Pennsylvania': [40.590752, -77.209755], 'Rhode Island': [41.680893, -71.511780],
    'South Carolina': [33.856892, -80.945007], 'South Dakota': [44.299782, -99.438828],
    'Tennessee': [35.747845, -86.692345], 'Texas': [31.054487, -97.563461],
    'Utah': [40.150032, -111.862434], 'Vermont': [44.045876, -72.710686],
    'Virginia': [37.769337, -78.169968], 'Washington': [47.400902, -121.490494],
    'West Virginia': [38.491226, -80.954570], 'Wisconsin': [44.268543, -89.616508],
    'Wyoming': [42.755966, -107.302490]
}
df["StateLat"] = df["State"].map(lambda x: state_coords.get(x, [None, None])[0])
df["StateLon"] = df["State"].map(lambda x: state_coords.get(x, [None, None])[1])

# ------------------- SIDEBAR FILTERS -------------------
with st.sidebar:
    st.title("Filters")

    sex_filter = st.multiselect("Sex", df["Sex"].unique(), default=df["Sex"].unique())
    age_filter = st.multiselect("Age Category", df["AgeCategory"].unique(), default=df["AgeCategory"].unique())
    race_filter = st.multiselect("Race / Ethnicity", df["RaceEthnicityCategory"].unique(), default=df["RaceEthnicityCategory"].unique())

    # Additional filters
    if "GeneralHealth" in df2.columns:
        health_filter = st.multiselect("General Health", df2["GeneralHealth"].unique(), default=df2["GeneralHealth"].unique())
    else:
        health_filter = []

    if "LastCheckupTime" in df2.columns:
        checkup_filter = st.multiselect("Last Checkup Time", df2["LastCheckupTime"].unique(), default=df2["LastCheckupTime"].unique())
    else:
        checkup_filter = []

# ------------------- FILTER DATA -------------------
df_filtered = df[
    df["Sex"].isin(sex_filter) &
    df["AgeCategory"].isin(age_filter) &
    df["RaceEthnicityCategory"].isin(race_filter)
]

# ------------------- METRICS -------------------
st.markdown("## 💡 Key Health Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    heart_attack_pct = df_filtered["HadHeartAttack"].str.lower().eq("yes").mean() * 100
    st.metric(label="❤️ Heart Attack %", value=f"{heart_attack_pct:.1f}%")

with col2:
    bmi_avg = df_filtered["BMI"].mean()
    st.metric(label="⚖️ Avg BMI", value=f"{bmi_avg:.1f}")

with col3:
    smoker_column_name = "SmokerStatus" 
    current_smoker_pct = df_filtered[smoker_column_name].str.lower().str.contains("current smoker").mean() * 100

    st.metric(label="🚬 Smokers %", value=f"{current_smoker_pct:.1f}%")

with col4:
    sleep_avg = df_filtered["SleepHours"].mean()
    st.metric(label="🛌 Avg Sleep", value=f"{sleep_avg:.1f} hrs")

# ------------------- NEW SECTION: Sleep Distribution -------------------
st.markdown("## 🛌 Sleep Distribution")
fig_sleep = ff.create_distplot([df_filtered["SleepHours"].dropna()], group_labels=["Sleep Hours"],
                               colors=["#60a5fa"], show_hist=False)
fig_sleep.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="white"))
st.plotly_chart(fig_sleep, use_container_width=True)

# ------------------- NEW SECTION: BMI vs Physical Health -------------------
st.markdown("## 🔍 BMI vs Physical Health")
fig_bmi = px.scatter(df_filtered, x="BMI", y="PhysicalHealthDays", color="Sex",
                     color_discrete_sequence=px.colors.sequential.Blues_r,
                     title="BMI vs Physical Health Days")
fig_bmi.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="white"))
st.plotly_chart(fig_bmi, use_container_width=True)

    
# ------------------- LINE CHART -------------------
st.markdown("## 📈 Physical & Mental Health Trends")
health_by_age = df_filtered.groupby("AgeCategory")[["PhysicalHealthDays", "MentalHealthDays"]].mean().reset_index()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=health_by_age["AgeCategory"], y=health_by_age["PhysicalHealthDays"],
    name="Physical Health", mode="lines+markers", line=dict(color="#22d3ee", width=3)
))
fig1.add_trace(go.Scatter(
    x=health_by_age["AgeCategory"], y=health_by_age["MentalHealthDays"],
    name="Mental Health", mode="lines+markers", line=dict(color="#8b5cf6", width=3)  # Purple tint
))

fig1.update_layout(
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font=dict(color="white", size=14),
    xaxis=dict(title="Age Category", showgrid=False),
    yaxis=dict(title="Avg Days", showgrid=True, gridcolor="#1e293b"),
    legend=dict(x=0.5, xanchor="center", orientation="h", y=1.1)
)
st.plotly_chart(fig1, use_container_width=True)

# ------------------- HISTOGRAM -------------------

st.markdown("## 🧑‍🤝‍🧑 Heart Attack Distribution by Sex")
fig2 = px.histogram(
    df_filtered,
    x="Sex",
    color="HadHeartAttack",
    barmode="group",
    color_discrete_map={"yes": "#22d3ee", "no": "#8b5cf6"},  
    title=None
)
fig2.update_layout(
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font=dict(color="white"),
    xaxis_title=None,
    yaxis_title="Count",
    legend_title="Heart Attack"
)
st.plotly_chart(fig2, use_container_width=True)


# ------------------- NEON STYLE MAP -------------------
st.markdown("## 🗺️ Heart Health by U.S. State")

# Filter dataset
state_attack = df_filtered[df_filtered["HadHeartAttack"].str.lower() == "yes"]
map_data = state_attack.groupby("State")[["StateLat", "StateLon"]].first().dropna()
map_data["HeartAttackRate"] = state_attack.groupby("State")["HadHeartAttack"].count()
map_data = map_data.reset_index()

# Normalize HeartAttackRate for color mapping
min_rate = map_data["HeartAttackRate"].min()
max_rate = map_data["HeartAttackRate"].max()

# Define function to map rate to blue shades
def rate_to_color(rate):
    def interpolate(start, end, factor):
        return int(start + (end - start) * factor)
    
    factor = (rate - min_rate) / (max_rate - min_rate) if max_rate > min_rate else 0
    r = interpolate(191, 0, factor)
    g = interpolate(242, 92, factor)
    b = interpolate(255, 160, factor)
    return [r, g, b, 200]

map_data["Color"] = map_data["HeartAttackRate"].apply(rate_to_color)

# PyDeck map layer
map_layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_data,
    get_position='[StateLon, StateLat]',
    get_radius=80000,
    get_fill_color="Color",
    pickable=True,
)

view_state = pdk.ViewState(
    longitude=-98, latitude=39, zoom=3.4, pitch=30
)


colorscale = [
    [0, 'rgb(191,242,255)'],
    [1, 'rgb(0,92,160)']
]

# Manually creating the colorbar to avoid unwanted blue block
colorbar = dict(
    title=dict(text="Heart Attack Rate", side="right"),
    tickvals=[0, 1],
    ticktext=[f"{int(min_rate)}", f"{int(max_rate)}"],
    thickness=15,
    len=0.7,
    ticks="outside",  # Ensure ticks are outside
    tickcolor="white",  # Color for ticks
    tickwidth=2         # Make tick width visible
)

# Layout for the map (remove the unwanted block)
col_map, col_legend = st.columns([4, 1])
with col_map:
    st.pydeck_chart(pdk.Deck(
        layers=[map_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10"
    ))

# Creating legend without unwanted blocks
with col_legend:
    # Custom plotly legend without unwanted light blue block
    fig_legend = go.Figure()

    fig_legend.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale=colorscale,
            cmin=min_rate,
            cmax=max_rate,
            color=[min_rate, max_rate],
            showscale=True,
            colorbar=colorbar
        ),
        showlegend=False
    ))

    fig_legend.update_layout(
        title="Heart Attack Rate",
        margin=dict(l=10, r=10, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    st.plotly_chart(fig_legend, use_container_width=True)

# ------------------- NEW SECTION: COVID Positivity by Health -------------------

if {'COVIDPositive', 'GeneralHealth'}.issubset(df2.columns):
    st.markdown("## 🦠 COVID Positivity by General Health")
    covid_health = df2.groupby(['GeneralHealth', 'COVIDPositive']).size().reset_index(name='Count')
    fig_covid = px.bar(covid_health, x='GeneralHealth', y='Count', color='COVIDPositive',
                       barmode='stack', color_discrete_map={'Yes': '#60a5fa', 'No': '#8b5cf6'})
    fig_covid.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="white"))
    st.plotly_chart(fig_covid, use_container_width=True)


# ------------------- SMOKERS PIE CHART -------------------
st.markdown("## 🚬 Smoking Status Distribution")

smoker_counts = df_filtered["SmokerStatus"].value_counts().reset_index()
smoker_counts.columns = ["SmokerStatus", "Count"]

fig_smokers = px.pie(
    smoker_counts,
    values="Count",
    names="SmokerStatus",
    color_discrete_sequence=["#0ea5e9", "#1e3a8a", "#3b82f6"]  
)

fig_smokers.update_traces(
    textinfo="percent+label"
)

fig_smokers.update_layout(
    paper_bgcolor="#0f172a",
    font=dict(color="white"),
)

st.plotly_chart(fig_smokers, use_container_width=True)



st.markdown("## 🚬 Heart Attacks by Smoking Status")
fig1 = px.histogram(df_filtered, x='SmokerStatus', color='HadHeartAttack', barmode='group',
                    title="Heart Attack Incidence by Smoking Status")
st.plotly_chart(fig1, use_container_width=True)


st.markdown("## ⚖️ BMI Distribution by Heart Attack Status")
fig2 = px.violin(df_filtered, y="BMI", color="HadHeartAttack", box=True, points="all",
                 title="BMI vs Heart Attack Occurrence")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("## 🏃 Physical Activity and Heart Attacks")
fig3 = px.histogram(df_filtered, x='PhysicalActivities', color='HadHeartAttack', barmode='group',
                    title="Heart Attacks vs Physical Activity")
st.plotly_chart(fig3, use_container_width=True)

st.markdown("## 🛌 Sleep Hours and Heart Attacks")
fig4 = px.box(df_filtered, x="HadHeartAttack", y="SleepHours", color="HadHeartAttack",
              title="Sleep Duration by Heart Attack History")
st.plotly_chart(fig4, use_container_width=True)


st.markdown("## 📊 Heart Attacks Across Age Groups")
fig5 = px.histogram(df_filtered, x='AgeCategory', color='HadHeartAttack', barmode='group',
                    title="Age vs Heart Attack")
st.plotly_chart(fig5, use_container_width=True)

st.markdown("## 🧠 Self-Reported Health and Heart Attacks")
fig6 = px.histogram(df_filtered, x='GeneralHealth', color='HadHeartAttack', barmode='group',
                    category_orders={"GeneralHealth": ["Excellent", "Very good", "Good", "Fair", "Poor"]},
                    title="General Health vs Heart Attack")
st.plotly_chart(fig6, use_container_width=True)

st.markdown("## 🩺 Medical Checkups and Heart Attacks")
fig7 = px.histogram(df_filtered, x='LastCheckupTime', color='HadHeartAttack', barmode='group',
                    title="Heart Attacks by Recency of Checkup")
st.plotly_chart(fig7, use_container_width=True)

st.markdown("## 🍬 Diabetes and Heart Attack Risk")
fig8 = px.histogram(df_filtered, x='HadDiabetes', color='HadHeartAttack', barmode='group',
                    title="Heart Attack Incidence by Diabetes Status")
st.plotly_chart(fig8, use_container_width=True)

st.markdown("## 📈 BMI Distribution by Age")
fig9 = px.box(
    df_filtered, x="AgeCategory", y="BMI", color="AgeCategory", title="BMI Across Age Categories",
    color_discrete_sequence=px.colors.sequential.Blues_r
)
fig9.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="white"))

st.plotly_chart(fig9, use_container_width=True)

st.markdown("## 🧠 Mental Health Days and Heart Attack Risk")
fig10 = px.box(df_filtered, x="HadHeartAttack", y="MentalHealthDays", color="HadHeartAttack",
               title="Mental Health Days vs Heart Attack")
st.plotly_chart(fig10, use_container_width=True)

st.markdown("## 🏥 General Health vs Heart Attack")

health_attack = df.groupby(['GeneralHealth', 'HadHeartAttack']).size().reset_index(name='Count')

fig_health = px.bar(health_attack, 
                    x='GeneralHealth', 
                    y='Count', 
                    color='HadHeartAttack', 
                    barmode='group',
                    title='Heart Attack by General Health Status')
st.plotly_chart(fig_health, use_container_width=True)


st.markdown("## 🦠 COVID-19 Positivity by General Health")

covid_health = df_filtered.groupby(['GeneralHealth', 'CovidPos']).size().reset_index(name='Count')

fig14 = px.bar(covid_health, 
               x='GeneralHealth', 
               y='Count', 
               color='CovidPos', 
               barmode='group',
               title='COVID-19 Positivity by General Health')
st.plotly_chart(fig14, use_container_width=True)

st.markdown("## 🧍‍♂️ BMI Distribution by Diabetes Status")

fig15 = px.box(df_filtered, 
               x='HadDiabetes', 
               y='BMI', 
               color='HadDiabetes',
               title='BMI Distribution for People With and Without Diabetes' , 
                color_discrete_sequence=px.colors.sequential.Blues_r)
fig15.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="white"))
st.plotly_chart(fig15, use_container_width=True)

st.markdown("## 📊 Correlation Heatmap of Numeric Features")

numeric_cols = ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours',
                'HeightInMeters', 'WeightInKilograms', 'BMI']

corr = df[numeric_cols].corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    title="Correlation Matrix (Numeric Features)",
    color_continuous_scale='ice'
)
fig_corr.update_layout(
    template='plotly_dark',
    width=1000,
    height=800,
    title_x=0.5,
    title_y=0.95
)
st.plotly_chart(fig_corr, use_container_width=True)


st.markdown("## 🛏️ Average Sleep Hours by Age Category")

sleep_age = df.groupby('AgeCategory')['SleepHours'].mean().reset_index()
fig_sleep = px.bar(
    sleep_age, 
    x='AgeCategory', 
    y='SleepHours', 
    title='Average Sleep Hours by Age Category',
    color='SleepHours',
    color_continuous_scale='ice'
)
fig_sleep.update_layout(template='plotly_dark')
st.plotly_chart(fig_sleep, use_container_width=True)
