from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Model


class CNN(Model):
    def __init__(self, input_dim, output_dim, input_length):
        self.model = Sequential()
        self.model.add(Embedding(input_dim = input_dim, output_dim = output_dim, input_length = input_length))
        self.model.add(Conv1D(128, 5, activation = 'relu'))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(10, activation = 'relu'))
        self.model.add(Dense(1, activation = 'sigmoid'))
    
    def compile(self, optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy']):
        self.model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    
    def summary(self):
        self.model.summary()

    def train_model(self, X_train, y_train, X_test, y_test, epochs):
        self.model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = epochs, batch_size = 32)
    
    def save(self, file_path):
        self.model.save(file_path)