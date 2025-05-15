def adaptive_loss(y_true, y_pred):  
    """Penalizes overconfidence (like Wood balancing Metal)."""  
    entropy = -np.sum(y_pred * np.log(y_pred))  
    rigidity_penalty = max(0, 0.2 - entropy)  # Punish low entropy  
    return keras.losses.categorical_crossentropy(y_true, y_pred) + rigidity_penalty  