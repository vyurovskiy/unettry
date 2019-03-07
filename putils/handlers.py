import torch
from tensorboardX import SummaryWriter as _SummaryWriter


class SummaryWriter(_SummaryWriter):
    def __init__(self, model, log_dir, shape):
        super().__init__(log_dir=log_dir)
        try:
            self.add_graph(model, torch.randn(2, *shape))
        except Exception as e:
            print(f"Failed to save model graph: {e}")

    def __del__(self):
        self.close()
