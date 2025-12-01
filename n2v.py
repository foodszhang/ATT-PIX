import matplotlib.pyplot as plt
import numpy as np
from careamics import CAREamist
from careamics.config import create_n2v_configuration
from careamics.utils.metrics import scale_invariant_psnr
from careamics_portfolio import PortfolioManager
from careamics.model_io import load_pretrained
from PIL import Image
import skimage as ski
import os

train_path = "./data/ECL"
result_path = "./result/ECL"
os.makedirs(result_path, exist_ok=True)
path_to_archive = "ECL_n2v_model.zip"
load_from_bmz = False
path_to_ckpt = "checkpoints/ECL-v2.ckpt"
load_from_ckpt = True

if __name__ == "__main__":
    # download file
    train_data = []
    index = 0
    index_map = {}
    careamist = None
    for file in os.listdir(train_path):
        if file.endswith(".tif"):
            train_data.append(ski.io.imread(os.path.join(train_path, file)))
            index_map[index] = file
            index += 1

    train_data = np.array(train_data)
    if not load_from_bmz and not load_from_ckpt:
        config = create_n2v_configuration(
            experiment_name="ECL",
            data_type="array",
            axes="SYX",
            patch_size=(64, 64),
            batch_size=4,
            num_epochs=10,
        )
        config.data_config.train_dataloader_params = {
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": True,
        }

        print(config)

        # instantiate a CAREamist
        # train
        careamist = CAREamist(source=config)
        careamist.train(
            train_source=train_data,
        )
    else:
        preload_path = path_to_ckpt if load_from_ckpt else path_to_archive
        careamist = CAREamist(source=preload_path)

    prediction = careamist.predict(source=train_data)
    for i, p in enumerate(prediction):
        filename = index_map[i]
        p = (p - p.min()) / (p.max() - p.min()) * 4095
        ski.io.imsave(os.path.join(result_path, filename), p.astype(np.int16))
    # Export the model
    careamist.export_to_bmz(
        path_to_archive=path_to_archive,
        friendly_model_name="ECL_N2V",
        input_array=train_data[0].astype(np.float32),
        authors=[{"name": "wenshuai zhou", "affiliation": ""}],
        data_description="ECL denoise",
        general_description="ECL denoise",
    )
