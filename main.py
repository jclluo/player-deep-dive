import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import percentileofscore
from pyscript import display, document # Interface with HTML

# --- GLOBAL VARIABLES ---
ml_df = None
master_df = None
scaler = None
kmeans = None
features = [
    'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 
    'FG_PCT', 'FG3_PCT', 'FT_PCT', 
    'FG3A', 'FGA', 'FTA'
]
cluster_names = {
    2: "All-Star / Primary Scorer",
    0: "Rotation Scorer / Wing",
    3: "Traditional Big / Rim Protector",
    1: "Deep Bench / Role Player"
}

# --- 1. LOAD DATA ---
def load_models():
    global ml_df, master_df, scaler, kmeans
    
    # In PyScript, files defined in 'fetch' are available locally
    ml_df = pd.read_pickle('model_data/ml_df.pkl')
    master_df = pd.read_pickle('model_data/master_season_pergame_df.pkl')
    
    with open('model_data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model_data/kmeans.pkl', 'rb') as f:
        kmeans = pickle.load(f)
        
    print("Models Loaded Successfully!")

# --- 2. HELPER FUNCTIONS ---
def get_percentile_ranks(player_stats, population_df, features):
    percentiles = []
    for feature, value in zip(features, player_stats):
        pct = percentileofscore(population_df[feature], value) / 100.0
        percentiles.append(pct)
    return percentiles

def create_radar_chart(player1_name, p1_percentiles, player2_name, p2_percentiles):
    from math import pi
    
    num_vars = len(features)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Styling for Dark Mode
    fig.patch.set_facecolor('#1e212b') 
    ax.set_facecolor('#1e212b')
    ax.spines['polar'].set_visible(False)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Labels
    plt.xticks(angles[:-1], features, color='white', size=8)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["", "", "", ""], color="grey", size=6)
    plt.ylim(0, 1.0)
    ax.grid(color='#444444', linestyle='--', linewidth=0.5)

    # Plot Data
    values1 = p1_percentiles + p1_percentiles[:1]
    ax.plot(angles, values1, linewidth=2, linestyle='solid', color='#00d4ff', label=player1_name)
    ax.fill(angles, values1, '#00d4ff', alpha=0.25)
    
    values2 = p2_percentiles + p2_percentiles[:1]
    ax.plot(angles, values2, linewidth=2, linestyle='solid', color='#ff2a6d', label=player2_name)
    ax.fill(angles, values2, '#ff2a6d', alpha=0.25)
    
    # Legend
    legend = plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='#1e212b', edgecolor='white')
    for text in legend.get_texts():
        text.set_color("white")
        
    return fig

# --- 3. MAIN EVENT LISTENER ---
def analyze_click(event):
    # 1. Get User Input from HTML
    input_name = document.querySelector("#playerInput").value
    
    # 2. Logic
    # Find Player ID from Name (Case Insensitive)
    player_row = master_df[master_df['PLAYER_NAME'].str.lower() == input_name.lower()]
    
    if player_row.empty:
        document.querySelector("#resultArea").innerHTML = f"<p style='color:red'>Player '{input_name}' not found.</p>"
        return

    # Get most recent season for simplicity or sort
    target_stats = player_row.sort_values('SEASON_ID', ascending=False).head(1)
    season_id = target_stats['SEASON_ID'].values[0]
    
    # Scale & Predict
    target_vector = target_stats[features].fillna(0)
    target_scaled = scaler.transform(target_vector)
    
    # Cluster
    pred_cluster = kmeans.predict(target_scaled)[0]
    archetype = cluster_names.get(pred_cluster, "Unknown")
    
    # Similarity
    distances = euclidean_distances(target_scaled, scaler.transform(ml_df[features].fillna(0)))
    closest_idx = distances[0].argmin()
    similarity_score = distances[0][closest_idx]
    
    modern_player = ml_df.iloc[closest_idx]
    modern_name = modern_player['PLAYER_NAME']
    
    # 3. Update HTML Results
    document.querySelector("#archetypeVal").innerText = archetype
    document.querySelector("#matchVal").innerText = modern_name
    document.querySelector("#scoreVal").innerText = f"{similarity_score:.3f}"
    
    # 4. Generate & Display Plot
    p1_pct = get_percentile_ranks(target_vector.values.flatten(), ml_df, features)
    p2_pct = get_percentile_ranks(modern_player[features].values.flatten(), ml_df, features)
    
    fig = create_radar_chart(target_stats['PLAYER_NAME'].values[0], p1_pct, modern_name, p2_pct)
    
    # Clear old plot and add new one
    plot_div = document.querySelector("#plotArea")
    plot_div.innerHTML = "" 
    display(fig, target="plotArea")

# --- INITIALIZATION ---
load_models()