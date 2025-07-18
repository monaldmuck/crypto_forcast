# train.py: Train the LSTM model
def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    # Fit the model on training data
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return history