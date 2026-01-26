import os

import torch
import yaml

from utility import Datasets
from models.MultiCBR import MultiCBR


def main():
    os.environ.setdefault("MULTICBR_QUIET_STATS", "1")

    conf = yaml.safe_load(open("./config.yaml", "r", encoding="utf-8"))["NetEase"]
    conf["dataset"] = "NetEase"
    conf["model"] = "MultiCBR"
    conf["device"] = torch.device("cpu")

    conf["batch_size_train"] = 8
    conf["batch_size_test"] = 8
    conf["num_workers_train"] = 0
    conf["num_workers_test"] = 0

    dataset = Datasets(conf)
    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items

    conf["embedding_size"] = conf["embedding_sizes"][0]
    conf["num_layers"] = conf["num_layerss"][0]
    conf["l2_reg"] = conf["l2_regs"][0]
    conf["UB_ratio"] = conf["UB_ratios"][0]
    conf["UI_ratio"] = conf["UI_ratios"][0]
    conf["BI_ratio"] = conf["BI_ratios"][0]
    conf["c_lambda"] = conf["c_lambdas"][0]
    conf["c_temp"] = conf["c_temps"][0]

    model = MultiCBR(conf, dataset.graphs).to(conf["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=conf["l2_reg"])

    model.train(True)
    users, bundles = next(iter(dataset.train_loader))
    users = users.to(conf["device"])
    bundles = bundles.to(conf["device"])
    bpr_loss, c_loss = model([users, bundles], ED_drop=False)
    loss = bpr_loss + conf["c_lambda"] * c_loss
    loss.backward()
    optimizer.step()

    print(f"smoke_ok loss={float(loss):.6f} bpr={float(bpr_loss):.6f} c={float(c_loss):.6f}")


if __name__ == "__main__":
    main()
