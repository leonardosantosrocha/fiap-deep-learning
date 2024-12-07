from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split

class LSTM_Custom:
    def __init__(self, n_layers=1, n_units=16, c_dropout_rate=0.1, c_input_shape=(16, 1), c_optimizer="adam", c_loss="binary_crossentropy"):
        self.n_layers = n_layers
        self.n_units = n_units
        self.c_dropout_rate = c_dropout_rate
        self.c_input_shape = c_input_shape
        self.c_optimizer = c_optimizer
        self.c_loss = c_loss
    

    def build_model(self):

        model = Sequential()

        # Se o número de camadas é menor ou igual a 1, logo a camada LSTM não poderá retornar os valores
        if self.n_layers <= 1:

            model.add(LSTM(units=self.n_units, return_sequences=False, input_shape=self.c_input_shape))
            model.add(Dropout(self.c_dropout_rate))
            model.add(Dense(units=1, activation="sigmoid"))
            
            model.compile(optimizer=self.c_optimizer, loss=self.c_loss, metrics=["accuracy"])
            
            return model

        # Se o número de camadas é maior que 1, logo todas as camadas LSTM (com exceção da última) poderão retornar os valores
        else: 
    
            for n_layer in range(self.n_layers):
                
                if n_layer == 0:
                    model.add(LSTM(units=self.n_units, return_sequences=True, input_shape=self.c_input_shape))
                    model.add(Dropout(self.c_dropout_rate))

                # Se o número da camada LSTM for diferente da última camada, logo a camada poderá retornar valores
                elif n_layer >= 1 and n_layer < self.n_layers - 1:
                    model.add(LSTM(units=self.n_units, return_sequences=True))
                    model.add(Dropout(self.c_dropout_rate * n_layer))
                
                # Se o número da camada LSTM for igual da última camada, logo a camada não poderá retornar valores
                else:
                    model.add(LSTM(units=self.n_units, return_sequences=False))
                    model.add(Dropout(self.c_dropout_rate * n_layer))
            
            model.add(Dense(units=1, activation="sigmoid"))
            
            model.compile(optimizer=self.c_optimizer, loss=self.c_loss, metrics=["accuracy"])

            return model
    

    def run_model(self, x_i, y_i, n_epochs, n_batch):

        x_train, x_test, y_train, y_test = train_test_split(x_i, y_i, test_size=0.30, random_state=42)

        model = self.build_model()

        statistics = model.fit(x_train, y_train, epochs=n_epochs, batch_size=n_batch, verbose=0)

        scores = model.evaluate(x_test, y_test)

        scores = [round(scores[0] * 100, 2), round(scores[1]* 100, 2)]

        return [scores, statistics.history["accuracy"], statistics.history["loss"]]