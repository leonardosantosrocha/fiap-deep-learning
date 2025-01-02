from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split

class Custom_LSTM:
    def __init__(self):
        self.model = None
        self.c_x_train = None
        self.c_y_train = None
        self.c_x_test = None
        self.c_y_test = None
        self.c_model_statistics_train = list()
        self.c_model_statistics_test = list()
        self.c_model_scores_train = list()
        self.c_model_scores_test = list()
        self.c_predicts = list()
    

    def build_model(self, c_layers_number=1, c_units_number=16, c_dropout_rate=0.1, c_input_shape=(16, 1)):
        self.model = Sequential()

        # Se o número de camadas é menor ou igual a 1, logo a camada LSTM não poderá retornar os valores
        if c_layers_number <= 1:

            self.model.add(LSTM(units=c_units_number, return_sequences=False, input_shape=c_input_shape))
            self.model.add(Dropout(c_dropout_rate))
            self.model.add(Dense(units=1, activation="sigmoid"))

        # Se o número de camadas é maior que 1, logo todas as camadas LSTM (com exceção da última) poderão retornar os valores
        else: 
    
            for layer_number in range(c_layers_number):
                
                if layer_number == 0:
                    self.model.add(LSTM(units=c_units_number, return_sequences=True, input_shape=c_input_shape))
                    self.model.add(Dropout(c_dropout_rate))

                # Se o número da camada LSTM for diferente da última camada, logo a camada poderá retornar valores
                elif layer_number >= 1 and layer_number < c_layers_number - 1:
                    self.model.add(LSTM(units=c_units_number, return_sequences=True))
                    self.model.add(Dropout(c_dropout_rate * layer_number))
                
                # Se o número da camada LSTM for igual da última camada, logo a camada não poderá retornar valores
                else:
                    self.model.add(LSTM(units=c_units_number, return_sequences=False))
                    self.model.add(Dropout(c_dropout_rate * layer_number))
            
            self.model.add(Dense(units=1, activation="sigmoid"))


    def compile_model(self, c_optimizer="adam", c_loss="binary_crossentropy", c_metrics=["accuracy"]):
        self.model.compile(optimizer=c_optimizer, loss=c_loss, metrics=c_metrics)


    def fit_model(self, c_x, c_y, c_test_size=0.3, c_random_state=42, c_epochs_number=100, c_batches_number=64, c_verbose=0):
        x_train, x_test, y_train, y_test = train_test_split(c_x, c_y, test_size=c_test_size, random_state=c_random_state)

        self.c_x_train = x_train
        self.c_y_train = y_train
        
        self.c_x_test = x_test
        self.c_y_test = y_test

        model_fitted = self.model.fit(x_train, y_train, epochs=c_epochs_number, batch_size=c_batches_number, verbose=c_verbose)

        self.c_model_statistics_train.append([model_fitted.history["accuracy"], model_fitted.history["loss"]])


    def predict_model(self, c_x):
        self.c_predicts.append(self.model.predict(c_x))


    def evaluate_model(self):
        model_evaluated = self.model.evaluate(self.c_x_test, self.c_y_test)

        self.c_model_scores_train.append(model_evaluated)