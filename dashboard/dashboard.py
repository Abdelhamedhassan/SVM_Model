import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import plotly.figure_factory as ff

# ------------------- PAGE SETUP -------------------
st.set_page_config(page_title="Heart Health Dashboard",page_icon='Heart-Disease.png', layout="wide")

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
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-radius: 16px;
        padding: 1rem !important;
        box-shadow: 0 0 12px rgba(96,165,250,0.4);
    }
    /* Filter Box Styling */
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
df = pd.read_csv("dashboard\heart_2022_no_nans.csv")

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

    with st.container():
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        sex_filter = st.multiselect("Sex", df["Sex"].unique(), default=df["Sex"].unique())
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        age_filter = st.multiselect("Age Category", df["AgeCategory"].unique(), default=df["AgeCategory"].unique())
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        race_filter = st.multiselect("Race / Ethnicity", df["RaceEthnicityCategory"].unique(), default=df["RaceEthnicityCategory"].unique())
        st.markdown("</div>", unsafe_allow_html=True)

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
    smoker_pct = df_filtered["SmokerStatus"].str.lower().eq("current smoker").mean() * 100
    st.metric(label="🚬 Smokers %", value=f"{smoker_pct:.1f}%")

with col4:
    sleep_avg = df_filtered["SleepHours"].mean()
    st.metric(label="🛌 Avg Sleep", value=f"{sleep_avg:.1f} hrs")

    
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
    color_discrete_map={"Yes": "#22d3ee", "No": "#8b5cf6"},  # Custom neon shades
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

# Plotly colorbar legend (using dummy heatmap)
colorscale = [
    [0, 'rgb(191,242,255)'],
    [1, 'rgb(0,92,160)']
]

legend_fig = ff.create_annotated_heatmap(
    z=[[0]],  # Use a real numeric value instead of None
    annotation_text=[[""]],
    colorscale=colorscale,
    showscale=True,
    colorbar=dict(
        title=dict(text="Heart Attack Rate", side="right"),
        tickvals=[0, 1],
        ticktext=[f"{int(min_rate)}", f"{int(max_rate)}"],
        thickness=15,
        len=0.7
    )
)

legend_fig.update_traces(
    showscale=True,
    hoverinfo='skip',
    xgap=0,
    ygap=0,
    colorscale=colorscale,
    zmin=0,
    zmax=1
)

legend_fig.update_layout(
    margin=dict(l=10, r=10, t=0, b=0),
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font=dict(color="white"),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False)
)

# Layout in Streamlit
col_map, col_legend = st.columns([4, 1])
with col_map:
    st.pydeck_chart(pdk.Deck(
        layers=[map_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10"
    ))

with col_legend:
    st.plotly_chart(legend_fig, use_container_width=True)
