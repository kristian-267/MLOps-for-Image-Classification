from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self, H, W):
        super().__init__()

        num_features = 1000
        in_chanels = [3, 4]
        out_chanels = [4, 5]
        conv_kernel = 3
        stride_kernel = 1
        padding_kernel = 1
        pool_kernel = 2
        dropout_p = 0.2
        num_conv = 2

        self.conv1 = nn.Conv2d(
            in_chanels[0],
            out_chanels[0],
            conv_kernel,
            stride=stride_kernel,
            padding=padding_kernel,
        )
        self.conv2 = nn.Conv2d(
            in_chanels[-1],
            out_chanels[-1],
            conv_kernel,
            stride=stride_kernel,
            padding=padding_kernel,
        )

        self.fc = nn.Linear(
            int(
                (H / pool_kernel**num_conv)
                * (W / pool_kernel**num_conv)
                * out_chanels[-1]
            ),
            num_features,
        )

        self.dropout = nn.Dropout(p=dropout_p)
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel)

        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.flatten = nn.Flatten(1, -1)

    def forward(self, x):
        x = self.pool(self.dropout(self.relu(self.conv1(x))))
        x = self.pool(self.dropout(self.relu(self.conv2(x))))
        x = self.logsoftmax(self.fc(self.flatten(x)))

        return x
