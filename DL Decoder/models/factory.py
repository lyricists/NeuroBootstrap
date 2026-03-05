from .classifiers_dl import EEGNet
from .classifiers_dl import ShallowConvNet
from .classifiers_dl import DeepConvNet
from .classifiers_dl import MLP


def get_dl_model(
    model_type: str,
    n_channels: int,
    n_samples: int,
    n_classes: int,
    input_dim: int | None = None,
):
    m = model_type.lower()
    if m == "eegnet":
        return EEGNet(n_channels, n_samples, n_classes)
    if m == "shallowconvnet":
        return ShallowConvNet(n_channels, n_samples, n_classes)
    if m == "deepconvnet":
        return DeepConvNet(n_channels, n_samples, n_classes)
    if m == "mlp":
        if input_dim is None:
            raise ValueError("MLP requires input_dim (n_channels * n_samples).")
        return MLP(input_dim=input_dim, n_classes=n_classes)
    raise ValueError(f"Unknown DL model: {model_type}")
