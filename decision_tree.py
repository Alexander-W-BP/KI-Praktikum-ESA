import numpy as np
from joblib import load
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt

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

# Pfad zur .viper-Datei
tree_path = "resources/viper_extracts/extract_output/Pong_seed0_reward-env_oc-extraction/Tree-16.1_best.viper"

# Entscheidungsbaum laden
tree = load(tree_path)

# Automatische Klassennamen generieren
classes = np.unique(tree.classes_)
class_names = [f"Class_{int(c)}" for c in classes]

# Entscheidungsbaum in Textform anzeigen
print("Entscheidungsbaum in Textform:")
print(export_text(tree, feature_names=feature_names))

# Grafische Darstellung
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=feature_names, class_names=class_names, filled=True)
plt.show()
