# 🏀 NBA Player DNA: Archetype & Similarity Engine

### **Project Overview**
This project is an end-to-end Data Science application that analyzes modern NBA player playstyles. By leveraging Machine Learning (K-Means Clustering) and Vector Space Modeling, the application identifies "Player Archetypes" and finds the closest statistical doppelgängers for historical legends in the modern game.


### **Technical Architecture**

#### **1. Data Collection (The Foundation)**
* **Source:** Official NBA Stats API (`nba_api`) and Basketball-Reference.
* **Scope:** Collected career data and individual season logs for over 4,500 players.
* **Datasets:** * `Master_Careers`: Aggregated career totals.
    * `Master_Seasons`: Granular per-season statistics.

#### **2. Data Preprocessing & Feature Engineering**
* **Normalization:** Converted raw totals to **Per-Game** statistics to account for varying games played.
* **Handling Nulls:** Imputed missing values for percentages (e.g., 0/0 3PA) to 0.0.
* **Standardization:** Applied `StandardScaler` (Z-Score Normalization) to ensure high-volume stats (Points) didn't overpower defensive metrics (Steals/Blocks) during distance calculations.

#### **3. Machine Learning Modeling**
* **Clustering (Unsupervised Learning):**
    * Algorithm: **K-Means Clustering**.
    * Optimization: Used the **Elbow Method** to determine the optimal number of clusters ($k=4$).
    * **Identified Archetypes:**
        1.  *Primary Ball Handlers / Stars* (High Usage, High AST)
        2.  *Rotation Wings / Scorers* (3&D profiles)
        3.  *Rim Protectors / Bigs* (High REB, High BLK)
        4.  *Deep Bench* (Low volume)
* **Similarity Engine:**
    * Algorithm: **Euclidean Distance** in $n$-dimensional feature space.
    * Functionality: Projects a historical player's vector onto the modern feature space to find the nearest neighbor (lowest distance).

#### **4. Visualization & Deployment**
* **Interactive App:** Built with **Streamlit** (Python).
* **Visuals:** Custom **Radar Charts** (Matplotlib) using percentile rankings to visualize playstyle differences.
* **Deployment:** Hosted on **Streamlit Community Cloud**.

---

### **Installation & Usage**

To run this project locally:

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/your-username/nba-player-dna.git](https://github.com/your-username/nba-player-dna.git)