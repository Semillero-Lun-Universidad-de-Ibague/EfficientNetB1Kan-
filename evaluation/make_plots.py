import json, argparse, time, re
import matplotlib.pyplot as plt
from datetime import date
from sklearn import metrics


def plot_confusion_matrix(true_labels: list, predictions: list, labels: list, model: str) -> None:
    
    cm = metrics.confusion_matrix(true_labels, predictions, labels=labels)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10,10))
    model_name = re.split('_\d+epochs', model)[0]
    plt.title('Confusion Matrix of Predicted Tumors for Model {}'.format(model_name))
    disp.plot(ax=ax)
    plt.savefig('plots/confusion_matrix_{}_plot{}.png'.format(model_name, date.today()))


def extract_preds(json_file: str, model_name: str) -> tuple[list, list]:

    with open(json_file, 'r') as jsn:
        preds_labels = json.load(jsn)

    for element in preds_labels:
        if model_name == element['model_name']:
            return (element['preds'], element['labels'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run this script in order to plot the confusion matrix of the desired model.')
    parser.add_argument('model_name', type=str,
                        help='pass the name of the desired model')
    parser.add_argument('json_name', type=str,
                        help='pass the name of the desired model')

    args = parser.parse_args()

    labels_list = list(range(4))
    preds, labels = extract_preds(args.json_name, args.model_name)

    print(len([y for i, y in enumerate(preds) if y == labels[i]]) / len(labels))

    plot_confusion_matrix(labels, preds, labels_list, args.model_name)