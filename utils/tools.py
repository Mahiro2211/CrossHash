import numpy as np
from torchvision import transforms
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
import torchvision.datasets as dsets
ImageFile.LOAD_TRUNCATED_IMAGES=True
from loguru import logger
import os
def config_dataset(config):
    if "cifar" in config["datasets"]:
        config["num_train"] = 10000
        config["num_query"] = 5000
        config["topK"] = -1
        config["n_class"] = 10
    elif config["datasets"] in ["nuswide_21", "nuswide_21_m"]:
        config["num_train"] = 10500
        config["num_query"] = 2100
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["datasets"] == "nuswide_81_m":
        config["num_train"] = 10000
        config["num_query"] = 5000
        config["topK"] = 5000
        config["n_class"] = 81
    elif config["datasets"] == "coco":
        config["num_train"] = 10000
        config["topK"] = -1
        config["num_query"] = 5000
        config["n_class"] = 80
    elif config["datasets"] == "imagenet":
        config["num_train"] = 13000
        config["num_query"] = 5000
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["datasets"] == "mirflickr" :
        config["num_train"] = 4000
        config["num_query"] = 1000
        config["topK"] = -1
        config["n_class"] = 38
    elif config["datasets"] == "UCMD":
        config["num_train"] = 1680
        config["num_query"] = 420
        config["topK"] = -1
        config["n_class"] = 17

    config["data_path"] = "./data/" + config["datasets"]
    if config["datasets"] == "cifar":
        config["data_path"] = "datasets/cifar"
    if config["datasets"] == "nuswide_21":
        config["data_path"] = "data/nuswide_v2_256"
    if config["datasets"] in ["nuswide_21_m", "nuswide_81_m"]:
        config["data_path"] = "datasets/nuswide_81"
    if config["datasets"] == "coco":
        config["data_path"] = "./data/coco"
    if config["datasets"] == "imagenet":
        config["data_path"] = "datasets/imagenet"
    if config["datasets"] == "mirflickr":
        config["data_path"] = "data/mirflickr"
    config["data"] = {
        "train_set": {"list_path": config["data_path"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": config["data_path"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": config["data_path"] + "/test.txt", "batch_size": config["batch_size"]}}

    return config

class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        # self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.imgs = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index


def cifar_dataset(config):
    batch_size = config["batch_size"]

    train_size = 500
    test_size = 100

    if config["datasets"] == "cifar10-2":
        train_size = 5000
        test_size = 1000

    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cifar_dataset_root = './datasets/'
    # Dataset
    train_dataset = MyCIFAR10(root=cifar_dataset_root,
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(root=cifar_dataset_root,
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR10(root=cifar_dataset_root,
                                 train=False,
                                 transform=transform)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if config["datasets"] == "cifar10":
        # test:1000, train:5000, database:54000
        pass
    elif config["datasets"] == "cifar10-1":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))
    elif config["datasets"] == "cifar10-2":
        # test:10000, train:50000, database:50000
        database_index = train_index

    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    logger.info("train_dataset: %d" %train_dataset.data.shape[0])
    logger.info("test_dataset: %d" %test_dataset.data.shape[0])
    logger.info("database_dataset: %d" %database_dataset.data.shape[0])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4)

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]


def get_data(config):
    if "cifar" in config["datasets"]:
        return cifar_dataset(config)

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        logger.info(data_set +" %d " %len(dsets[data_set]))
        if data_set == "train_set":
            dset_loaders[data_set] = torch.utils.data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle=True, num_workers=4,drop_last=False)
        else:
            dset_loaders[data_set] = torch.utils.data.DataLoader(dsets[data_set],
                                                                 batch_size=data_config[data_set]["batch_size"],
                                                                 shuffle=False, num_workers=4,drop_last=False)

    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])


def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader, ncols=60):
        with torch.amp.autocast(device_type="cuda",dtype=torch.float16):
            clses.append(cls)
            x = net(img.to(device))
            bs.append((x).data.cpu())
    net.train()
    return torch.cat(bs).sign(), torch.cat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk, topimg):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query),ncols=40):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        index = ind[0:topimg]
        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap,index


# faster but more memory
def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk):
    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec = np.zeros((num_query, num_gallery))
    recall = np.zeros((num_query, num_gallery))
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        all_sim_num = np.sum(gnd)

        prec_sum = np.cumsum(gnd)
        return_images = np.arange(1, num_gallery + 1)

        prec[iter, :] = prec_sum / return_images
        recall[iter, :] = prec_sum / all_sim_num

        assert recall[iter, -1] == 1.0
        assert all_sim_num == prec_sum[-1]

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    index = np.argwhere(recall[:, -1] == 1.0)
    index = index.squeeze()
    prec = prec[index]
    recall = recall[index]
    cum_prec = np.mean(prec, 0)
    cum_recall = np.mean(recall, 0)

    return topkmap, cum_prec, cum_recall

from save_mat import Save_mat
# https://github.com/chrisbyd/DeepHash-pytorch/blob/master/validate.py
def validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_image,):
    device = config["test_device"]
    net = net.to(device)

    tst_binary, tst_label = compute_result(test_loader, net, device=device)

    trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

    mAP, index = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), config["topK"],config["top_img"])
    Save_mat(epoch=epoch, output_dim=bit, datasets=config['datasets'], query_labels=tst_label, retrieval_labels=trn_label,
             query_img=tst_binary, retrieval_img=trn_binary, save_dir='.',
             mode_name=config['info'],map=mAP)
    dataset = config['datasets']

    if mAP > Best_mAP:
        Best_mAP = mAP
        # if "save_path" in config:
        #     save_path = os.path.join(config["save_path"], f'{config["datasets"]}_{bit}bits_{mAP}')
        #     os.makedirs(save_path, exist_ok=True)
        #     print("save in ", save_path)
        #     np.save(os.path.join(save_path, "tst_label.npy"), tst_label.numpy())
        #     np.save(os.path.join(save_path, "tst_binary.npy"), tst_binary.numpy())
        #     np.save(os.path.join(save_path, "trn_binary.npy"), trn_binary.numpy())
        #     np.save(os.path.join(save_path, "trn_label.npy"), trn_label.numpy())
        # os.makedirs(f'./hole_state_dict_{dataset}',exist_ok=True)
        os.makedirs(f'./state_dict_{dataset}',exist_ok=True)
        torch.save(net.state_dict(), os.path.join(f'./state_dict_{dataset}', f'{Best_mAP}_{bit}_model.pth'))
        # torch.save(net, os.path.join(f'./hole_state_dict_{dataset}', f'{Best_mAP}_{bit}_model.pth'))
    logger.info(f"{config['info']} epoch:{epoch + 1} bit:{bit} datasets:{config['datasets']} MAP:{mAP} Best MAP: {Best_mAP}")
    # logger.info("The serial number of the first ten images returned:" + index)
    return Best_mAP, index
