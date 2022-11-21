## Part of codes are referred from: 
## https://github.com/peterwestai2/symbolic-knowledge-distillation/blob/main/purification_code/predict.py

import os
import sys
import argparse
import pandas as pd
import sklearn.metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_result_file", type=str)
    parser.add_argument("--recalls", nargs="+", type=float, default=[.5, .6, .7, .8, .9])
    parser.add_argument("--do_plot", action="store_true", help="plot the precison-recall curve")

    args = parser.parse_args()
    res = pd.read_csv(args.dev_result_file, sep="\t")

    for ts in [0.5, 0.6, 0.7, 0.8, 0.9]: 

        res["new_pred"] = res["score"].map(lambda x: "v" if x >= ts else "i")
        acc = accuracy_score(res.label, res["new_pred"])
        precision = precision_score(res.label, res["new_pred"], pos_label="v")
        recall = recall_score(res.label, res["new_pred"], pos_label="v")
        f1 = f1_score(res.label, res["new_pred"], pos_label="v")
        print("The threshold at {}: acc {:.5f}; precision {:.5f}; recall {:.5f}; f1 {:.5f}".format(ts, acc, precision, recall, f1))

    val_label = res.label
    val_pred = res.score

    val_ps, val_rs, val_thresh = sklearn.metrics.precision_recall_curve(y_true=val_label, 
                                                                    probas_pred=val_pred,
                                                                    pos_label="v")
    if args.do_plot:
        import matplotlib.pyplot as plt

        plt.figure(1) 
        plt.title('Precision/Recall Curve')
        plt.ylabel('Precision')
        plt.xlabel('Recall')

        plt.plot(val_ps, val_rs)
        plt.show()
        
        file_ext = args.dev_result_file.split("/")[-1].split(".")[0]
        plt.savefig(file_ext + "_prcurve.png")

    for recall in args.recalls:
        idx = 0 
        while val_rs[idx] > recall:
            idx +=1
        print('Val precision@{:.0f}%: {:.3f},  threshold={:.5f}'.format(recall*100, val_ps[idx], val_thresh[idx]))


if __name__== "__main__":
    main()
