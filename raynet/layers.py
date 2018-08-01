from keras import backend as K
from keras import initializers
from keras.engine.topology import Layer


class LayerNormalization(Layer):
    """LayerNormalization is a determenistic normalization layer to replace
    BN's stochasticity.

    # Arguments
        axes: list of axes that won't be aggregated over
    """
    def __init__(self, axes=None, bias_axis=-1, epsilon=1e-3, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.axes = axes
        self.bias_axis = bias_axis
        self.epsilon = epsilon

    def build(self, input_shape):
        # Get the number of dimensions and the axes that won't be aggregated
        # over
        ndims = len(input_shape)
        dims = input_shape[self.bias_axis]
        axes = self.axes or []

        # Figure out the shape of the statistics
        shape = [1]*ndims
        for ax in axes:
            shape[ax] = input_shape[ax]

        # Figure out the axes we will aggregate over accounting for negative
        # axes
        self.reduction_axes = [
            ax for ax in range(ndims)
            if ax > 0 and (ax+ndims) % ndims not in axes
        ]

        # Create trainable variables
        self.gamma = self.add_weight(
            shape=shape,
            name="gamma",
            initializer=initializers.get("ones")
        )
        self.bias = self.add_weight(
            shape=(dims,),
            name="bias",
            initializer=initializers.get("zeros")
        )

        self.built = True

    def call(self, inputs):
        x = inputs
        assert not isinstance(x, list)

        # Compute the per sample statistics
        mean = K.mean(x, self.reduction_axes, keepdims=True)
        std = K.std(x, self.reduction_axes, keepdims=True) + self.epsilon

        return self.gamma*(x-mean)/std + self.bias


class CustomSoftmax(Layer):
    def __init__(self, **kwargs):
        super(CustomSoftmax, self).__init__(**kwargs)

    def call(self, inputs):
        X = inputs[0]
        N = inputs[1]

        t_xshape = K.shape(X)    # xshape as tensor
        xshape = K.int_shape(X)  # xshape as ints

        # Create a mask that keeps only N values for each sample and zeroes the
        # rest
        mask = K.map_fn(
            lambda i: K.arange(xshape[1]) < N[i, 0],
            K.arange(0, t_xshape[0]),
            dtype="bool"
        )
        mask = K.reshape(mask, t_xshape)
        mask = K.cast(mask, "float32")

        # Compute the softmax of the inputs ignoring the unused inputs
        max_X = K.map_fn(
            lambda i: K.max(X[i, :N[i, 0]]),
            K.arange(0, t_xshape[0]),
            dtype="float32"
        )
        max_X = K.reshape(max_X, (-1, 1))
        exp_X = K.exp(X - max_X)
        sum_exp_X = K.map_fn(
            lambda i: K.sum(exp_X[i, :N[i, 0]]),
            K.arange(0, t_xshape[0]),
            dtype="float32"
        )
        sum_exp_X = K.reshape(sum_exp_X, (-1, 1))

        return mask * exp_X / sum_exp_X
