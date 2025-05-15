from core.model import SelfHealingModel
from core.adaptivity import heal_dataset
import numpy as np

# Generate drifting data
X = np.concatenate([np.random.normal(0, 1, (500, 5)), 
                    np.random.normal(1, 2, (500, 5))])  # Drift halfway!
y = np.random.randint(0, 2, 1000)

# Heal + Train
X_healed, y_healed = heal_dataset(X, y)
model = SelfHealingModel(input_dim=5)
model.fit(X_healed, y_healed)  # Works! (Placeholder)