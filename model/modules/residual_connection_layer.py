import torch.nn as nn


class ResidualConnectionLayer(nn.Module):

    def __init__(self, norm, dr_rate=0):
        super(ResidualConnectionLayer, self).__init__()
        self.norm = norm
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, x, sub_layer):
        out = x
        out = self.norm(out)
        out = sub_layer(out)
        out = self.dropout(out)
        out = out + x
        return out


class IdentityResidualLayer(nn.Module):
    """
    It does residual connection without any trainable parameters (e.g. nn.LayerNorm, nn.Dropout, etc.).
    """
    def __init__(self):
        super(IdentityResidualLayer, self).__init__()

    def forward(self, x, sub_layer, *args, **kwargs):
        """ GPT says (may need fact-checking):
        The `IdentityResidualLayer` does not contain any trainable parameters or stateful components, and
        its forward method is essentially a static function. While it's somewhat unusual to have a nn.Module subclass
        with no parameters, it's NOT incorrect or problematic.

        A nn.Module in PyTorch doesn't strictly need to contain trainable parameters. A module can simply be a holder
        for a particular operation or set of operations, parameterized or not. In fact, PyTorch provides several
        such modules in its torch.nn library, such as nn.ReLU, nn.Sigmoid, nn.Softmax, nn.Dropout, and nn.Identity,
        none of which have trainable parameters.

        While creating a class for a static function might seem overkill, one of the benefits of doing so in PyTorch is
        that it allows you to seamlessly integrate these operations into your module hierarchy. It lets you call
        .to(device), .eval(), .train(), etc. on your entire model and have these methods propagate through
        all submodules correctly.

        However, please note that nn.Module subclasses with no parameters will not be picked up by
        torchsummary.summary() or similar methods to display a model's architecture and number of parameters.
        This is because these methods only consider layers with trainable parameters.

        As long as you are aware of this, using nn.Module subclasses with no parameters is perfectly fine
        and can sometimes make your code more organized and easier to manage.
        """
        return x + sub_layer(x, *args, **kwargs)


class NoResidualButSameForward(nn.Module):
    """
    It does not do a residual connection, but it does the same forward pass as the sub_layer.
    It also does the normalization if norm is not None or nn.Identity().
    """
    def __init__(self, norm):
        super(NoResidualButSameForward, self).__init__()
        self.norm = norm
        # Update dropout layer if necessary

    def forward(self, x, sub_layer, *args, **kwargs):
        # Just to align with the grammar of the placeholder where this class is replaced
        if self.norm is not None:
            x = self.norm(x)
        return sub_layer(x, *args, **kwargs)
