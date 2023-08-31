import numpy as np
import pandas as pd
from sklearn.metrics import *

def save_classification(y_true,y_pred, out_dir, labels):
  def accuracy(y_true, y_pred):
    numerator = np.sum(y_true & y_pred, axis=1)
    denominator = np.sum(y_true | y_pred, axis=1)
    n = y_true.shape[0]
    return np.sum(numerator / denominator) / n

  out = classification_report(y_true,y_pred, output_dict=True, target_names=labels)
  total_support = out['samples avg']['support']

  mr = accuracy_score(y_true, y_pred)
  acc = accuracy(y_true,y_pred)
  hm = hamming_loss(y_true, y_pred)

  out['Exact Match Ratio'] = {'precision': mr, 'recall': mr, 'f1-score': mr, 'support': total_support}
  out['Hamming Loss'] = {'precision': hm, 'recall': hm, 'f1-score': hm, 'support': total_support}
  out['Accuracy'] = {'precision': acc, 'recall': acc, 'f1-score': acc, 'support': total_support}
  out_df = pd.DataFrame(out).transpose()
  print(out_df)

  out_df.to_csv(out_dir)

  return out_df