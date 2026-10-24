import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE

def spliting_data(X, y, RANDOM_STATE=RANDOM_STATE):
      """Sépare les données en train, test et validation comme dans le notebook."""

      X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

      X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5,  stratify=y_tmp, random_state=RANDOM_STATE)

      print(X_train.shape, X_test.shape, X_val.shape)
      print(y_train.shape, y_test.shape, y_val.shape)

      return X_train, X_test, X_val, y_train, y_test, y_val

