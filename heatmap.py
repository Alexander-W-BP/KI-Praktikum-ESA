from joblib import load
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

# Pfad zur .viper-Datei
#tree_path = "resources/viper_extracts/extract_output/Pong_seed0_reward-env_oc-extraction/Tree-16.1_best.viper"
#tree_path = "resources/viper_extracts/extract_output/LunarLander_seed0_reward-mixed_oc-n25-extraction/Tree--362.17902_best.viper"
#tree_path = "resources/viper_extracts/extract_output/LunarLander_seed0_reward-mixed_oc-n26-extraction/Tree--559.71783_best.viper"
#tree_path = "resources/viper_extracts/extract_output/LunarLander_seed0_reward-human_oc-n3-extraction/Tree--333.47455_best.viper"
#tree_path = "resources/viper_extracts/extract_output/LunarLander_seed0_reward-human_oc-n4-extraction/Tree--308.82727_best.viper"
#tree_path = "resources/viper_extracts/extract_output/LunarLander_seed0_reward-human_oc-n10-extraction/Tree--348.6018_best.viper"
tree_path = "resources/viper_extracts/extract_output/LunarLander_seed0_reward-env_oc-n15-extraction/Tree--128.24423_best.viper"
#tree_path = "resources/viper_extracts/extract_output/Pong_seed0_reward-env_oc-n9-extraction/Tree-21.0_best.viper"

# Entscheidungsbaum laden
tree: DecisionTreeClassifier = load(tree_path)

# Feature-Wichtigkeiten extrahieren
feature_importances = tree.feature_importances_

# Liste der Features
feature_names = ['Player1.x', 'Player1.y', 'Enemy1.x', 'Enemy1.y', 'Ball1.x', 'Ball1.y', 
                 'Player1.x', 'Player1.y', 'Player1.x[t-1]', 'Player1.y[t-1]', 'Enemy1.x', 
                 'Enemy1.y', 'Enemy1.x[t-1]', 'Enemy1.y[t-1]', 'Ball1.x', 'Ball1.y', 
                 'Ball1.x[t-1]', 'Ball1.y[t-1]', 'O(Player1)', 'O(Enemy1)', 'O(Ball1)', 
                 'RGB(Player1.R)', 'RGB(Player1.G)', 'RGB(Player1.B)', 'RGB(Enemy1.R)', 
                 'RGB(Enemy1.G)', 'RGB(Enemy1.B)', 'RGB(Ball1.R)', 'RGB(Ball1.G)', 
                 'RGB(Ball1.B)', 'LT(Player1, Player1).x', 'LT(Player1, Player1).y', 
                 'LT(Player1, Enemy1).x', 'LT(Player1, Enemy1).y', 'LT(Player1, Ball1).x', 
                 'LT(Player1, Ball1).y', 'LT(Enemy1, Player1).x', 'LT(Enemy1, Player1).y', 
                 'LT(Enemy1, Enemy1).x', 'LT(Enemy1, Enemy1).y', 'LT(Enemy1, Ball1).x', 
                 'LT(Enemy1, Ball1).y', 'LT(Ball1, Player1).x', 'LT(Ball1, Player1).y', 
                 'LT(Ball1, Enemy1).x', 'LT(Ball1, Enemy1).y', 'LT(Ball1, Ball1).x', 
                 'LT(Ball1, Ball1).y', 'D(Player1, Enemy1).x', 'D(Player1, Enemy1).y', 
                 'D(Player1, Ball1).x', 'D(Player1, Ball1).y', 'D(Enemy1, Player1).x', 
                 'D(Enemy1, Player1).y', 'D(Enemy1, Ball1).x', 'D(Enemy1, Ball1).y', 
                 'D(Ball1, Player1).x', 'D(Ball1, Player1).y', 'D(Ball1, Enemy1).x', 
                 'D(Ball1, Enemy1).y', 'ED(Player1, Enemy1)', 'ED(Player1, Ball1)', 
                 'ED(Enemy1, Player1)', 'ED(Enemy1, Ball1)', 'ED(Ball1, Player1)', 
                 'ED(Ball1, Enemy1)', 'C(Player1, Enemy1).x', 'C(Player1, Enemy1).y', 
                 'C(Player1, Ball1).x', 'C(Player1, Ball1).y', 'C(Enemy1, Player1).x', 
                 'C(Enemy1, Player1).y', 'C(Enemy1, Ball1).x', 'C(Enemy1, Ball1).y', 
                 'C(Ball1, Player1).x', 'C(Ball1, Player1).y', 'C(Ball1, Enemy1).x', 
                 'C(Ball1, Enemy1).y', 'V(Player1).x', 'V(Enemy1).x', 'V(Ball1).x', 
                 'DV(Player1).x', 'DV(Player1).y', 'DV(Enemy1).x', 'DV(Enemy1).y', 
                 'DV(Ball1).x', 'DV(Ball1).y', 'COL(Player1)', 'COL(Enemy1)', 'COL(Ball1)']

feature_names_lunarlander = ['lander.x', 'lander.y', 'legs_1.x', 'legs_1.y', 'legs_2.x', 'legs_2.y', 'moon.x', 'moon.y', 'lander.x', 'lander.y', 'lander.x[t-1]', 'lander.y[t-1]', 'legs_1.x', 'legs_1.y', 'legs_1.x[t-1]', 'legs_1.y[t-1]', 'legs_2.x', 'legs_2.y', 'legs_2.x[t-1]', 'legs_2.y[t-1]', 'moon.x', 'moon.y', 'moon.x[t-1]', 'moon.y[t-1]', 'O(lander)', 'O(legs_1)', 'O(legs_2)', 'O(moon)', 'RGB(lander.R)', 'RGB(lander.G)', 'RGB(lander.B)', 'RGB(legs_1.R)', 'RGB(legs_1.G)', 'RGB(legs_1.B)', 'RGB(legs_2.R)', 'RGB(legs_2.G)', 'RGB(legs_2.B)', 'RGB(moon.R)', 'RGB(moon.G)', 'RGB(moon.B)', 'LT(lander, lander).x', 'LT(lander, lander).y', 'D(lander, moon).x', 'D(lander, moon).y', 'D(lander, legs_1).x', 'D(lander, legs_1).y', 'D(lander, legs_2).x', 'D(lander, legs_2).y', 'ED(lander, moon)', 'ED(lander, legs_1)', 'ED(lander, legs_2)', 'C(lander, moon).x', 'C(lander, moon).y', 'C(lander, legs_1).x', 'C(lander, legs_1).y', 'C(lander, legs_2).x', 'C(lander, legs_2).y', 'V(lander).x', 'DV(lander).x', 'DV(lander).y', 'COL(lander)', 'COL(legs_1)', 'COL(legs_2)', 'COL(moon)']

importance_matrix = np.reshape(feature_importances, (1, -1))

plt.figure(figsize=(20, 10))
sns.heatmap(importance_matrix, annot=True, cmap='viridis', xticklabels=feature_names_lunarlander, yticklabels=['Importance'])
plt.title('Feature Importances Heatmap')
plt.show()
plt.savefig('feature_importances_heatmap_Lunar_Lander_Agent_n26.png')