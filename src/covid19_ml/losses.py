import torch
import torch.nn as nn


class WeightedMSE_MAE_LOG_Loss(nn.Module):
    def __init__(self, use_horizon_weighting=False):
        super(WeightedMSE_MAE_LOG_Loss, self).__init__()
        self.use_horizon_weighting = use_horizon_weighting

    def forward(self, input, target):
        length = input.shape[0]
        return self._get_weighted_mse_loss(
            input, target, length, self.use_horizon_weighting
        )

    @staticmethod
    def symmetric_log(x, epsilon=1e-8):
        return torch.sign(x) * torch.log(torch.abs(x) + 1)

    def _get_weighted_mse_loss(
        self,
        input,
        target,
        length,
        use_horizon_weighting=False,
    ):
        if use_horizon_weighting:
            # Create a weight tensor of shape (n)
            weight = torch.tensor(
                [1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5], dtype=torch.float32
            )
        else:
            # Create a weight tensor of shape (n)
            weight = torch.ones(length, dtype=torch.float32)

        if input.device != weight.device:
            weight = weight.to(
                input.device
            )  # Ensuring the weight tensor is on the same device as input tensor

        mse_loss = nn.functional.mse_loss(input, target, reduction="none")
        mae_loss = nn.functional.l1_loss(input, target, reduction="none")
        log_loss = (self.symmetric_log(input) - self.symmetric_log(target)) ** 2

        return ((mse_loss + mae_loss + log_loss) * weight).mean()


class WeightedMSELoss(nn.Module):
    def __init__(self, use_horizon_weighting=False):
        super(WeightedMSELoss, self).__init__()
        self.use_horizon_weighting = use_horizon_weighting

    def forward(self, input, target):
        mse_loss = torch.nn.functional.mse_loss(input, target, reduction="none")
        n_items = input.shape[0]
        weights = torch.ones(n_items)
        if self.use_horizon_weighting:
            weights *= torch.linspace(1, n_items / 2, n_items)
        return mse_loss * weights


class WeightedRelativeMSELoss(nn.Module):
    def __init__(self, use_horizon_weighting=False):
        super(WeightedRelativeMSELoss, self).__init__()
        self.use_horizon_weighting = use_horizon_weighting

    def forward(self, input, target):
        mse_loss = ((input - target) / (target + 1e-3)) ** 2
        n_items = input.shape[0]
        weights = torch.ones(n_items)
        if self.use_horizon_weighting:
            weights *= torch.linspace(1, n_items / 2, n_items)
        return mse_loss * weights


class WeightedRelativeMAELoss(nn.Module):
    def __init__(self, use_horizon_weighting=False):
        super(WeightedRelativeMAELoss, self).__init__()
        self.use_horizon_weighting = use_horizon_weighting

    def forward(self, input, target):
        mae_loss = ((input - target) / (target + 1e-3)).abs()
        n_items = input.shape[0]
        weights = torch.ones(n_items)
        if self.use_horizon_weighting:
            weights *= torch.linspace(1, n_items / 2, n_items)
        return mae_loss * weights


class WeightedMAELoss(nn.Module):
    def __init__(self, use_horizon_weighting=False):
        super(WeightedMAELoss, self).__init__()
        self.use_horizon_weighting = use_horizon_weighting

    def forward(self, input, target):
        mae_loss = (input - target).abs()
        n_items = input.shape[0]
        weights = torch.ones(n_items)
        if self.use_horizon_weighting:
            weights *= torch.linspace(1, n_items / 2, n_items)
        return mae_loss * weights


class WeightedLogLoss(nn.Module):
    def __init__(self, use_horizon_weighting=False):
        super(WeightedLogLoss, self).__init__()
        self.use_horizon_weighting = use_horizon_weighting

    @staticmethod
    def symmetric_log(x):
        return torch.log10(x + 1)

    def forward(self, input, target):
        log_loss = self.symmetric_log(torch.abs(input - target))
        n_items = input.shape[0]
        weights = torch.ones(n_items)
        if self.use_horizon_weighting:
            weights *= torch.linspace(1, n_items / 2, n_items)
        return log_loss * weights


class WeightedRelativeLogLoss(nn.Module):
    def __init__(self, use_horizon_weighting=False):
        super(WeightedRelativeLogLoss, self).__init__()
        self.use_horizon_weighting = use_horizon_weighting

    @staticmethod
    def symmetric_log(x):
        return torch.log10(x + 1)

    def forward(self, input, target):
        log_loss = self.symmetric_log(torch.abs((input - target) / (target + 1e-3)))
        n_items = input.shape[0]
        weights = torch.ones(n_items)
        if self.use_horizon_weighting:
            weights *= torch.linspace(1, n_items / 2, n_items)
        return log_loss * weights


class MSEMAELogLoss(nn.Module):
    def __init__(self, use_horizon_weighting=False):
        super(MSEMAELogLoss, self).__init__()
        self.use_horizon_weighting = use_horizon_weighting
        self.mse = WeightedMSELoss(False)
        self.mae = WeightedMAELoss(False)
        self.log = WeightedLogLoss(False)

    @staticmethod
    def symmetric_log(x):
        return torch.log10(x + 1)

    def forward(self, input, target):
        n_items = input.shape[0]
        weights = torch.ones(n_items)
        if self.use_horizon_weighting:
            weights *= torch.linspace(1, n_items / 2, n_items)
        log_loss = self.log(input, target)
        mse_loss = self.mse(input, target)
        mae_loss = self.mae(input, target)
        return ((log_loss + mse_loss + mae_loss) / 3) * weights


class MSEMAELogRelativeLoss(nn.Module):
    def __init__(self, use_horizon_weighting=False):
        super(MSEMAELogRelativeLoss, self).__init__()
        self.use_horizon_weighting = use_horizon_weighting
        self.mse = WeightedRelativeMSELoss(False)
        self.mae = WeightedRelativeMAELoss(False)
        self.log = WeightedRelativeLogLoss(False)

    @staticmethod
    def symmetric_log(x):
        return torch.log10(x + 1)

    def forward(self, input, target):
        n_items = input.shape[0]
        weights = torch.ones(n_items)
        if self.use_horizon_weighting:
            weights *= torch.linspace(1, n_items / 2, n_items)
        log_loss = self.log(input, target)
        mse_loss = self.mse(input, target)
        mae_loss = self.mae(input, target)
        return ((log_loss + mse_loss + mae_loss) / 3) * weights


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        return nn.functional.cross_entropy(input, target)


class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, input_decoder_output, target_label_probabilities):
        """Expects input to be log probabilities and target to be probabilities."""
        input = torch.nn.functional.log_softmax(input_decoder_output, dim=1)

        return nn.functional.kl_div(
            input, target_label_probabilities, reduction="batchmean"
        )


LOSSES_DICO = {
    "MSE": WeightedMSELoss,
    "MAE": WeightedMAELoss,
    "MLP1AE": WeightedLogLoss,
    "MSSE": WeightedRelativeMSELoss,
    "MASE": WeightedRelativeMAELoss,
    "MLP1ASE": WeightedRelativeLogLoss,
    "MSE + MAE + MLP1AE": MSEMAELogLoss,
    "MSSE + MASE + MLP1ASE": MSEMAELogRelativeLoss,
    "old school": WeightedMSE_MAE_LOG_Loss,
}
if __name__ == "__main__":

    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    # Configure warnings to be raised as exceptions

    pio.templates.default = "plotly_white"

    def loss_lines():
        y_true = torch.ones(201).reshape(-1, 1)
        y_hat = torch.pow(torch.tensor([10]), torch.linspace(-1, 1, 201)).reshape(-1, 1)
        fig = make_subplots(1, 3, horizontal_spacing=0.1)
        for i, (name, loss_class) in enumerate(LOSSES_DICO.items()):
            diff = y_hat - y_true
            losses_val = []
            loss_fun = loss_class(use_horizon_weighting=True)
            for j in range(y_true.shape[0]):
                loss_val = loss_fun(y_hat[j], y_true[j])
                losses_val.append(loss_val)
            losses_val = torch.tensor(losses_val)

            # Plot Linear
            fig.add_trace(
                go.Scatter(
                    x=diff.numpy().flatten(),
                    y=losses_val.numpy(),
                    mode="lines",
                    # name=f"{name} Linear",
                ),
                row=1,
                col=i + 1,
            )

            # Plot Proportional
            # fig.add_trace(
            #     go.Scatter(
            #         x=diff.numpy().flatten(),
            #         y=losses_val.numpy(),
            #         mode="lines",
            #         # name=f"{name} Log",
            #     ),
            #     row=2,
            #     col=1,
            # )
            fig.update_layout(
                height=500,
                width=550,
                title_text=name,
                showlegend=False,
                # template="presentation",
            )
            fig.update_yaxes(row=1, col=i + 1, title="Loss value")
            # fig.update_yaxes(row=2, col=1, title="Loss value")
            fig.update_xaxes(row=1, col=i + 1, title="Error")
            # fig.update_xaxes(type="log", row=2, col=1, title="Error (log10)")
            # fig.write_image(f"{name}.png")
        fig.show()

    def loss_heatmaps():
        true_vals = torch.linspace(-1000, 1000, 201).reshape(-1, 1)
        errors = torch.linspace(-1000, 1000, 201).reshape(-1, 1)

        for i, (name, loss_class) in enumerate(LOSSES_DICO.items()):
            loss_fun = loss_class(use_horizon_weighting=True)
            losses_matrix = torch.zeros(len(true_vals), len(errors))

            for j in range(len(true_vals)):
                for k in range(len(errors)):
                    y_true = true_vals[j]
                    y_hat = true_vals[j] + errors[k]
                    loss_val = loss_fun(y_hat, y_true)
                    losses_matrix[j, k] = loss_val
            if name in ["MSSE", "MASE", "MLP1ASE", "MSSE + MASE + MLP1ASE"]:
                losses_matrix[99:102, :] = torch.max(losses_matrix[102:])
            fig = go.Figure(
                data=go.Heatmap(
                    z=losses_matrix.numpy().T,
                    y=errors.numpy().flatten(),
                    x=true_vals.numpy().flatten(),
                    colorscale="Magma",
                    colorbar=dict(title="Loss Value"),
                )
            )

            fig.update_layout(
                title=f"{name} Loss Heatmap",
                yaxis_title="Error Size",
                xaxis_title="True Value Size",
                height=800,
                width=800,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1,
                    xanchor="right",
                    x=1,
                ),
                template="presentation",
            )
            fig.write_image(f"{name}_heatmap.png")
            fig.show()

    loss_lines()
