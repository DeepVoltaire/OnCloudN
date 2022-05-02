import segmentation_models_pytorch as smp


def build_model(hps):
    """Builds a PyTorch segmentation model.
    Builds a PyTorch segmentation model according to the type of architecture, encoder backbone,
    number of input channel and classes specified.
    Args:
        hps (Hyperparams): Hyperparameters of the model to be built.
    Raises:
        NotImplementedError: The model architecture is not implemented.
    Returns:
        nn.Module: PyTorch segmentation model ready for inference
    """

    num_classes = hps.num_classes + 1
    backbone = hps.backbone if hps.backbone[:4] != "timm" else hps.backbone.replace("_", "-")
    encoder_weights = "imagenet" if hps.backbone[:4] != "timm" else "noisy-student"

    if hps.model == "unet":
        decoder_channels = [int(x * hps.smp_decoder_channels_mult) for x in [256, 128, 64, 32, 16]]
        decoder_attention_type = None if hps.smp_decoder_use_attention == 0 else "scse"
        print(
            hps.model,
            backbone,
            encoder_weights,
            f"BatchNorm: {hps.smp_decoder_use_batchnorm==1}",
            f"Decoder Channel: {decoder_channels}, Decoder Attention Type: {decoder_attention_type}",
        )
        net = smp.Unet(
            encoder_name=backbone,
            encoder_depth=5,
            encoder_weights=encoder_weights,
            decoder_use_batchnorm=hps.smp_decoder_use_batchnorm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=hps.input_channel,
            classes=num_classes,
        )
    elif hps.model == "unetplusplus":
        decoder_channels = [int(x * hps.smp_decoder_channels_mult) for x in [256, 128, 64, 32, 16]]
        decoder_attention_type = None if hps.smp_decoder_use_attention == 0 else "scse"
        print(
            hps.model,
            backbone,
            encoder_weights,
            f"BatchNorm: {hps.smp_decoder_use_batchnorm==1}",
            f"Decoder Channel: {decoder_channels}, Decoder Attention Type: {decoder_attention_type}",
        )
        net = smp.UnetPlusPlus(
            encoder_name=backbone,
            encoder_depth=5,
            encoder_weights=encoder_weights,
            decoder_use_batchnorm=hps.smp_decoder_use_batchnorm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=hps.input_channel,
            classes=num_classes,
        )
    else:
        raise NotImplementedError(hps.model)

    return net
