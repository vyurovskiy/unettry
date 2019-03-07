from models.unet import load_model
from putils.data import gen_test_data
from putils.generator import TestDataset


def main(model_path):
    model = load_model(3, 1, model_path)


if __name__ == '__main__':
    main(model_path)
