import keras.backend as K

from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Cropping2D, Activation
from keras.layers.merge import multiply
from keras.initializers import he_normal


if K.image_data_format() == "channels_first":
    _AXIS = 1
else:
    _AXIS = 3


def get_fm_shape(incomming):
    """
    Computes the feature map's shape, excluding the batch size.
    
    Parameters
    ----------
    incomming: tensor
        Tensor for which we need the shape.
    
    Returns
    -------
    tuple
        The shape of the tensor (channel, height, width)
    """
    incomming_shape = incomming._keras_shape
    if K.image_data_format() == "channels_first":
        ch, h, w = incomming_shape[1], incomming_shape[2], incomming_shape[3]
    else:
        ch, h, w = incomming_shape[3], incomming_shape[1], incomming_shape[2]

    return ch, h, w


def get_crop_from_output(incomming, output_shape):
    """
    Computes the ammount of crop from each side of the feature maps in order 
    to meet the desired shape.
    
    Parameters
    ----------
    incomming: tensor
        Tensor for which we need to calculate the amount of cropping.
    output_shape: list
        List of length 2 with the desired shape
    
    Returns
    -------
    tuple.
        The amount of shape to crop from each side of each dimension.
    """
    ch, h, w = get_fm_shape(incomming=incomming)
    input_shape = (h, w)

    assert len(input_shape) == len(output_shape)
    for i in range(len(input_shape)):
        assert input_shape[i] % 2 == 0
        assert output_shape[i] % 2 == 0

    total_crop = [(i_s - o_s) // 2 for i_s,
                  o_s in zip(input_shape, output_shape)]

    return tuple((crop, crop) for crop in total_crop)


def segse(incomming, dilation_rate, compression_ratio=10,
          kernel_regularizer=None, padding='valid'):
    """
    SegSE block. Feature maps' recalibration for semantinc segmentation with 
    FCN.

    A conv. layer with 3X3 kernels with dilation aggregates context. At the 
    same time, the feature maps are compressed. Then, a conv. layer with 1X1 
    kernels restores the number of feature maps, and the sigmoid activation is 
    applied element-wise. Finally, it is multiplied by the original input.
    
    Further description can be found in the paper.
    
    Parameters
    ----------
    incomming: tensor
        Input tensor, which is the output of the previous layer.
    dilation_rate: int
        Dilation rate to be used in the Conv. layer with 3X3 kernels.
    compression_ratio: int
        Compression ratio. The number of feature maps is divided by this 
        ratio in the Conv. layer with 3X3 kernels.
    kernel_regularizer: float
        Weight decay applied to the convolutional layers.
    padding: string
        Padding type: valid or same.
    
    Returns
    -------
    tensor.
        The recalibrated feature maps.
    """
    ch, h, w = get_fm_shape(incomming=incomming)

    init = incomming

    x = Conv2D(ch // compression_ratio, 3, padding=padding,
               dilation_rate=dilation_rate,
               kernel_regularizer=kernel_regularizer,
               kernel_initializer=he_normal())(init)
    x = BatchNormalization(axis=_AXIS)(x)
    x = Activation('relu')(x)
    x = Conv2D(ch, 1, padding=padding,
               kernel_initializer=he_normal(),
               kernel_regularizer=kernel_regularizer)(x)
    x = Activation('sigmoid')(x)

    if padding == 'valid':
        cropping = get_crop_from_output(incomming=incomming,
                                        output_shape=[h - dilation_rate * 2,
                                                      w - dilation_rate * 2])
        incomming = Cropping2D(cropping=cropping)(incomming)

    return multiply([incomming, x])


def recombination(incomming, expansion_factor=4, kernel_regularizer=None):
    """
    Recombination block.

    Linear recombination of the feature maps by expansion, followed by 
    compression to the original number of feature maps. Both are accomplished 
    by conv. layers with 1X1 kernels.
    
    Further description can be found in the paper.
    
    Parameters
    ----------
    incomming: tensor
        Input tensor, which is the output of the previous layer.
    expansion_factor: int
        Factor by which to expand the number of feature maps.
    kernel_regularizer: float
        Weight decay applied to the convolutional layers.
    
    Returns
    -------
    tensor.
        The recombined feature maps.
    """
    ch, h, w = get_fm_shape(incomming=incomming)
    exp_filters = ch * expansion_factor

    x = Conv2D(exp_filters, 1, kernel_initializer=he_normal(),
               kernel_regularizer=kernel_regularizer)(incomming)
    x = Conv2D(ch, 1, kernel_initializer=he_normal(),
               kernel_regularizer=kernel_regularizer)(x)

    return x


def recombination_and_recalibration(incomming, dilation_rate,
                                    compression_ratio=10, expansion_factor=4,
                                    kernel_regularizer=None, padding='valid'):
    """
    Recombination and recalibration block. This corresponds to the RR SegSE
    block in the paper.

    We first expand the feature maps. Then, we feed it to the SegSE block. 
    Finally, the original number of feature maps is recovered by compression.
    
    Further description can be found in the paper.
    
    Parameters
    ----------
    incomming: tensor
        Input tensor, which is the output of the previous layer.
    dilation_rate: int
        Dilation rate to be used in the Conv. layer with 3X3 kernels.
    compression_ratio: int
        Compression ratio. The number of feature maps is divided by this 
        ratio in the Conv. layer with 3X3 kernels.
    expansion_factor: int
        Factor by which to expand the number of feature maps.
    kernel_regularizer: float
        Weight decay applied to the convolutional layers.
    padding: string
        Padding type: valid or same.
    
    Returns
    -------
    tensor.
        The recombined and recalibrated feature maps.
    """
    ch, h, w = get_fm_shape(incomming=incomming)
    exp_filters = ch * expansion_factor

    # recombination - expansion
    x = Conv2D(exp_filters, 1, kernel_initializer=he_normal(),
               kernel_regularizer=kernel_regularizer)(incomming)

    # recalibration with SegSE
    x = segse(incomming=x, dilation_rate=dilation_rate,
              compression_ratio=compression_ratio,
              kernel_regularizer=kernel_regularizer, padding=padding)

    # recombination - compression
    x = Conv2D(ch, 1, kernel_initializer=he_normal(),
               kernel_regularizer=kernel_regularizer)(x)

    return x
