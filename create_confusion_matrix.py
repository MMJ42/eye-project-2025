import os
import json
import argparse
import sys

import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from utils import read_split_data,read_test_data
from my_dataset import MyDataSet
from model import swin_base_patch4_window7_224 as create_model
import csv

class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):

        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)


        table = PrettyTable()
        table.field_names = ["", "Accuracy", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Accuracy = round((TP + TN) / (TP + TN + FN + FP), 4) if TP + TN + FN + FP != 0 else 0.
            Precision = round(TP / (TP + FP), 4) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 4) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 4) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Accuracy, Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)


        plt.xticks(range(self.num_classes), self.labels, rotation=45)

        plt.yticks(range(self.num_classes), self.labels)

        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):

                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    test_rate = 1

    val_in_trainval_rate = test_rate
    val_images_path, val_images_label = read_test_data(args.data_path,val_in_trainval_rate)


    data_transform = {
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes)

    assert os.path.exists(args.weights), "cannot find {} file".format(args.weights)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)


    json_label_path = './class_indices_8label.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    predict_label_list = []
    true_label_list = []

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=args.num_classes, labels=labels)
    model.eval()
    with torch.no_grad():
        for val_data in tqdm(val_loader, file=sys.stdout):
            val_images, val_labels = val_data
            outputs0 = model(val_images.to(device))
            outputs1 = torch.softmax(outputs0, dim=1)
            outputs = torch.argmax(outputs1, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())

            predict_label_list.append(np.array(outputs1[0].to("cpu")))
            true_label_list.append(np.array(val_labels.to("cpu").numpy()))
    confusion.plot()
    confusion.summary()
    save_pred_and_true = 0
    if save_pred_and_true:
        header = ["AMD","CNV","CSR","DME","DR","DRUSEN","MH","NORMAL"]
        with open('predict_label.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(header)

            writer.writerows(predict_label_list)
        header = ['true_label']
        with open('true_label.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(header)

            writer.writerows(true_label_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=1)

    parser.add_argument('--data-path', type=str,
                        default="D:\\zhichao_workstation\\TTSH_Image_Classification\\OCT_mingjie\\OCT_combined_dataset_3channel_71515_8label\\Test")


    parser.add_argument('--weights', type=str, default='all_extenal_data_weights/model.pth',
                        help='initial weights path')

    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
