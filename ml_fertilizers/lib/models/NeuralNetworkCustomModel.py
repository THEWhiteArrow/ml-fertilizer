import keras
import numpy as np
from keras.api.layers import (
    Conv1D,
    SimpleRNN,
    Input,
    LSTM,
    GRU,
    Dense,
    Dropout,
    BatchNormalization,
)
from keras.api.models import Model
from keras.api.optimizers import Adam
from sklearn.base import TransformerMixin
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from ml_fertilizers.lib.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class NeuralNetworkCustomModel(TransformerMixin):

    n_layers: int = 2
    layer_type: str = "dense"
    units: int = 64
    activation: str = "relu"
    dropout: float = 0.1
    has_batch_norm: bool = False
    learning_rate: float = 0.001
    input_size: Optional[int] = None
    epochs: int = 10
    batch_size: int = 32
    validation_split: float = 0.2
    positive_class_threshold: float = 0.5
    callbacks: Optional[List] = None
    window: Optional[int] = 1
    verbose: int = 1
    task_type: str = "classification"

    def __post_init__(self):
        self.model = None

    def fit(self, X, y) -> "NeuralNetworkCustomModel":
        self.input_size = X.shape[1]
        self.create_model()

        X_fit = self.create_sequences(X, self.window)
        y_fit = y.iloc[self.window - 1 :]

        self.model.fit(
            X_fit,
            y_fit,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=self.callbacks,
            verbose=self.verbose,
        )

        del X_fit
        del X

        return self

    def predict(self, X) -> np.ndarray:

        if len(X) < self.window:
            raise ValueError(
                f"Input data length {len(X)} is less than the window size {self.window}."
            )

        X_fit = self.create_sequences(X, self.window)

        y_pred_raw = self.model.predict(X_fit)
        y_pred = np.where(
            y_pred_raw > self.positive_class_threshold,
            1.0,
            0.0,
        )

        del X_fit
        del X

        return y_pred

    def __clone__(self) -> "NeuralNetworkCustomModel":
        return self.create_model()

    @staticmethod
    def create_sequences(data: np.ndarray, window: int):
        """
        Create sequences of length `window` from the input data.

        Args:
            data (np.ndarray): The input data of shape (num_samples, num_features).
            window (int): The sequence length.

        Returns:
            np.ndarray: The reshaped data of shape (num_sequences, window, num_features).
        """
        sequences = []
        # if data is sparse then convert to dense, create series and convert back to sparse if possible
        if hasattr(data, "toarray"):
            data = data.toarray()
        for i in range(len(data) - window + 1):
            sequences.append(data[i : i + window])
        del data
        return np.array(sequences)

    def create_model(self) -> "NeuralNetworkCustomModel":
        if self.model is not None:
            del self.model
            keras.backend.clear_session()

        if self.input_size is None:
            logger.warning("Skipping model creation because input_size is None.")
            return self

        inputs = Input(shape=(self.window, self.input_size))
        x = inputs
        for i in range(self.n_layers):
            if self.layer_type == "gru":
                x = GRU(self.units, activation=self.activation, return_sequences=True)(
                    x
                )
            elif self.layer_type == "lstm":
                x = LSTM(self.units, activation=self.activation, return_sequences=True)(
                    x
                )
            elif self.layer_type == "dense":
                x = Dense(self.units, activation=self.activation)(x)

            elif self.layer_type == "rnn":
                x = SimpleRNN(
                    self.units, activation=self.activation, return_sequences=True
                )(x)
            elif self.layer_type == "conv":
                x = Conv1D(
                    filters=self.units,
                    kernel_size=3,
                    activation=self.activation,
                    padding="same",
                )(x)

            if self.has_batch_norm:
                x = BatchNormalization()(x)

            if self.dropout > 0:
                x = Dropout(self.dropout)(x)

        x = LSTM(32, activation="tanh", return_sequences=False)(x)
        x = Dropout(self.dropout)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(self.dropout)(x)

        outputs = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inputs, outputs=outputs)

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy" if self.task_type == "classification" else "mse",
            metrics=["accuracy"] if self.task_type == "classification" else ["mse"],
        )
        self.model = model
        return self

    def set_output(self, *, transform=None):
        return self

    def set_params(self, **params) -> "NeuralNetworkCustomModel":
        """
        layer_type
        units
        activation
        dropout
        batch_norm
        learning_rate
        inputs
        epochs
        batch_size
        validation_split
        """
        self.n_layers = params.get("n_layers", self.n_layers)
        self.layer_type = params.get("layer_type", self.layer_type)
        self.units = params.get("units", self.units)
        self.activation = params.get("activation", self.activation)
        self.dropout = params.get("dropout", self.dropout)
        self.has_batch_norm = params.get("batch_norm", self.has_batch_norm)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.input_size = params.get("input_size", self.input_size)
        self.epochs = params.get("epochs", self.epochs)
        self.batch_size = params.get("batch_size", self.batch_size)
        self.validation_split = params.get("validation_split", self.validation_split)
        self.positive_class_threshold = params.get(
            "positive_class_threshold", self.positive_class_threshold
        )
        self.window = params.get("window", self.window)
        self.callbacks = params.get("callbacks", self.callbacks)
        self.verbose = params.get("verbose", self.verbose)

        return self.create_model()

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return asdict(self)
