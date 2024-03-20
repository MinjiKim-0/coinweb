import csv
from os import write


def save_to_file(filename, days, y_preds):
    file = open(filename, mode="w")
    writer = csv.writer(file)
    writer.writerow(["Date", "Price"])
    for i in range(30):
        writer.writerow([days[i], y_preds[i]])
    return
