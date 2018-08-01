import os

import h5py
import numpy as np

from keras import backend as K
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dot, \
    Dropout, Flatten, Input, MaxPooling2D, GlobalAveragePooling2D
from keras import regularizers
from keras.layers import Average as KerasAverage
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.engine.topology import Layer

from .layers import LayerNormalization, CustomSoftmax
from .tf_implementations.loss_functions import loss_factory


class TotalReshape(Layer):
    def __init__(self, target_shape, **kwargs):
        self.target_shape = target_shape
        super(TotalReshape, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return tuple(
            x if x != -1 else None
            for x in self.target_shape
        )

    def call(self, x):
        return K.reshape(x, self.target_shape)


class BaseReducer(Layer):
    def __init__(self, **kwargs):
        super(BaseReducer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Average(BaseReducer):
    def call(self, x):
        return K.mean(x, axis=-1)


class Max(BaseReducer):
    def call(self, x):
        return K.max(x, axis=-1)


class TopKAverage(BaseReducer):
    def __init__(self, k, **kwargs):
        self.k = k
        super(TopKAverage, self).__init__(**kwargs)

    def call(self, x):
        if K.backend() == "tensorflow":
            tf = K.tf
            x, _ = tf.nn.top_k(x, self.k, sorted=False)
            return K.mean(x, axis=-1)
        else:
            raise NotImplementedError("TopKAverage is not implemented for "
                                      " %s backend" % (K.backend(),))


def reducer_factory(reducer, k=3):
    # Set the type of the reducer to be used
    if reducer == "max":
        return Max()
    elif reducer == "average":
        return Average()
    elif reducer == "topK":
        return TopKAverage(k)


def mae(y_true, y_pred):
    """ Implementation of Mean average error
    """
    return K.mean(K.abs(y_true - y_pred))


def mde(y_true, y_pred):
    return K.mean(K.cast(
        K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1)),
        K.floatx()
    ))


def create_simple_cnn(input_shape, kernel_regularizer=None):
    common_params = dict(
        filters=32,
        kernel_size=3,
        kernel_regularizer=kernel_regularizer
    )
    return Sequential([
        Conv2D(input_shape=input_shape, **common_params),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(**common_params),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(**common_params),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(**common_params),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(**common_params),
        BatchNormalization()
    ])


def create_simple_cnn_ln(input_shape, kernel_regularizer=None):
    common_params = dict(
        filters=32,
        kernel_size=3,
        kernel_regularizer=kernel_regularizer
    )
    return Sequential([
        Conv2D(input_shape=input_shape, **common_params),
        LayerNormalization(),
        Activation("relu"),
        Conv2D(**common_params),
        LayerNormalization(),
        Activation("relu"),
        Conv2D(**common_params),
        LayerNormalization(),
        Activation("relu"),
        Conv2D(**common_params),
        LayerNormalization(),
        Activation("relu"),
        Conv2D(**common_params),
        LayerNormalization()
    ])


def create_dilated_cnn_receptive_field_25(
    input_shape,
    kernel_regularizer=None
):
    return Sequential([
        Conv2D(
            filters=32,
            kernel_size=5,
            input_shape=input_shape,
            kernel_regularizer=kernel_regularizer
        ),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(
            filters=32,
            kernel_size=5,
            kernel_regularizer=kernel_regularizer
        ),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(
            filters=32,
            kernel_size=5,
            kernel_regularizer=kernel_regularizer,
            dilation_rate=2
        ),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(
            filters=32,
            kernel_size=3,
            kernel_regularizer=kernel_regularizer,
        ),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(
            filters=32,
            kernel_size=3,
            kernel_regularizer=kernel_regularizer
        ),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(
            filters=32,
            kernel_size=3,
            kernel_regularizer=kernel_regularizer
        ),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(
            filters=32,
            kernel_size=3,
            kernel_regularizer=kernel_regularizer
        ),
        BatchNormalization()
    ])


def create_dilated_cnn_receptive_field_25_with_tanh(
    input_shape,
    kernel_regularizer=None
):
    return Sequential([
        Conv2D(
            filters=32,
            kernel_size=5,
            input_shape=input_shape,
            kernel_regularizer=kernel_regularizer
        ),
        BatchNormalization(),
        Activation("tanh"),
        Conv2D(
            filters=32,
            kernel_size=5,
            kernel_regularizer=kernel_regularizer
        ),
        BatchNormalization(),
        Activation("tanh"),
        Conv2D(
            filters=32,
            kernel_size=5,
            kernel_regularizer=kernel_regularizer,
            dilation_rate=2
        ),
        BatchNormalization(),
        Activation("tanh"),
        Conv2D(
            filters=32,
            kernel_size=3,
            kernel_regularizer=kernel_regularizer,
        ),
        BatchNormalization(),
        Activation("tanh"),
        Conv2D(
            filters=32,
            kernel_size=3,
            kernel_regularizer=kernel_regularizer
        ),
        BatchNormalization(),
        Activation("tanh"),
        Conv2D(
            filters=32,
            kernel_size=3,
            kernel_regularizer=kernel_regularizer
        ),
        BatchNormalization(),
        Activation("tanh"),
        Conv2D(
            filters=32,
            kernel_size=3,
            kernel_regularizer=kernel_regularizer
        ),
        BatchNormalization()
    ])


def create_hartmann_cnn(input_shape, kernel_regularizer=None):
    return Sequential([
        Conv2D(filters=32, kernel_size=5, input_shape=input_shape),
        Activation("tanh"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=5),
        Activation("tanh"),
        MaxPooling2D(pool_size=(2, 2))
    ])


def cnn_factory(name):
    cnn_factories = {
        "simple_cnn": create_simple_cnn,
        "simple_cnn_ln": create_simple_cnn_ln,
        "dilated_cnn_receptive_field_25":
            create_dilated_cnn_receptive_field_25,
        "dilated_cnn_receptive_field_25_with_tanh":
            create_dilated_cnn_receptive_field_25_with_tanh,
        "hartmann_cnn": create_hartmann_cnn
    }
    return cnn_factories[name]


def optimizer_factory(optimizer, lr, momentum=None, clipnorm=0.0, clipvalue=1):
    # Set the type of optimizer to be used
    if optimizer == "Adam":
        return Adam(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue)
    elif optimizer == "SGD":
        return SGD(lr=lr, momentum=momentum, clipnorm=clipnorm,
                   clipvalue=clipvalue)


def kernel_regularizer_factory(regularizer_factor):
    if regularizer_factor == 0.0:
        return None
    else:
        return regularizers.l2(regularizer_factor)


def build_simple_cnn(
    input_shape,
    create_cnn,
    optimizer="Adam",
    lr=1e-3,
    momentum=None,
    clipnorm=0.0,
    loss="mse",
    reducer="average",
    merge_layer="dot-product",
    weight_decay=None,
    weight_file=None
):
    # Make sure that we have a proper input shape
    # TODO: Maybe change this to 3, because we finally need only the
    # patch_shape?
    assert len(input_shape) == 5

    # Unpack the input shape to make the code more readable
    D, N, W, H, C = input_shape

    model = create_cnn(
        input_shape=(None, None, C),
        kernel_regularizer=weight_decay
    )
    model.compile(
        optimizer=optimizer_factory(
            optimizer,
            lr=lr,
            momentum=momentum,
            clipnorm=clipnorm
        ),
        loss=loss_factory(loss)
    )

    # If there is a weight file specified load the weights
    if weight_file:
        try:
            f = h5py.File(weight_file, "r")
            keys = [os.path.join(model.name, w.name)
                    for l in model.layers for w in l.weights]
            weights = [f[os.path.join("model_weights", k)][:] for k in keys]

            model.set_weights(weights)
        except:
            model.load_weights(weight_file, by_name=True)

    return model


def build_simple_nn_for_training(
    input_shape,
    create_cnn,
    optimizer="Adam",
    lr=1e-3,
    momentum=None,
    clipnorm=0.0,
    loss="emd",
    reducer="average",
    merge_layer="dot-product",
    weight_decay=None,
    weight_file=None
):
    # Make sure that we have a proper input shape
    assert len(input_shape) == 5

    # Unpack the input shape to make the code more readable
    D, N, W, H, C = input_shape

    # Create the two stream inputs
    x1_in = Input(shape=input_shape)
    x2_in = Input(shape=input_shape)

    # Reshape them for input in the CNN
    x1 = TotalReshape((-1, W, H, C))(x1_in)
    x2 = TotalReshape((-1, W, H, C))(x2_in)

    # Create the CNN and extract features from both streams
    cnn = create_cnn(input_shape=(W, H, C), kernel_regularizer=weight_decay)
    x1 = Flatten()(cnn(x1))
    x2 = Flatten()(cnn(x2))

    # Compute a kind of similarity between the features of the two streams
    x = Dot(axes=-1, normalize=(merge_layer == "cosine-similarity"))([x1, x2])

    # Reshape them back into their semantic shape (depth planes, patches, etc)
    x = TotalReshape((-1, D, N))(x)

    # Compute the final similarity scores for each depth plane
    x = reducer_factory(reducer)(x)

    # Compute the final output
    y = Activation("softmax")(x)

    model = Model(inputs=[x1_in, x2_in], outputs=y)
    model.compile(
        optimizer=optimizer_factory(
            optimizer,
            lr=lr,
            momentum=momentum,
            clipnorm=clipnorm
        ),
        loss=loss_factory(loss),
        metrics=["accuracy", mae, mde]
    )

    if weight_file:
        model.load_weights(weight_file, by_name=True)

    return model


def build_hartmann_network(
    input_shape,
    create_cnn=create_hartmann_cnn,
    optimizer="SGD",
    lr=1e-3,
    momentum=None,
    clipnorm=0.0,
    loss=None,
    reducer=None,
    merge_layer=None,
    weight_decay=None,
    weight_file=None
):
    # Make sure that we have a proper input shape
    assert len(input_shape) == 3

    # Unpack the input shape to make the code more readable
    H, W, C = input_shape

    # Create the feature extracting CNN
    cnn = create_hartmann_cnn(input_shape=(None, None, C))

    # Create the similarity CNN
    sim = Sequential([
        Conv2D(
            filters=2048,
            kernel_size=5,
            input_shape=K.int_shape(cnn.output)[1:]
        ),
        Activation("relu"),
        Conv2D(filters=2048, kernel_size=1),
        Activation("relu"),
        Conv2D(filters=2, kernel_size=1),
        Activation("softmax")
    ])

    # Create the joint model for training
    x_in = [Input(shape=input_shape) for i in range(5)]
    x = [cnn(xi) for xi in x_in]
    x = KerasAverage()(x)
    y = sim(x)
    model = Model(inputs=x_in, outputs=y)

    # Compile all the models
    model.compile(
        optimizer=optimizer_factory(
            optimizer,
            lr=lr,
            momentum=momentum,
            clipnorm=clipnorm
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    cnn.compile("sgd", "mse")  # Just so that we can run predict()
    sim.compile("sgd", "mse")

    # Attach the cnn and sim to the model in case someone wants to use them
    model.cnn = cnn
    model.sim = sim

    if weight_file:
        model.load_weights(weight_file, by_name=True)

    return model


def get_nn(name):
    models = {
        "simple_cnn": build_simple_cnn,
        "simple_nn_for_training": build_simple_nn_for_training,
        "hartmann": build_hartmann_network
    }
    return models[name]
