import numpy as np
import tqdm as tqdm
from scipy.special import logsumexp
from keras.datasets.mnist import load_data

class MLP():

  def __init__(self,din,dout):
    self.W=(2*np.random.rand(dout,din)-1)*(np.sqrt(6)/np.sqrt(din*dout))
    self.b=(2*np.random.rand(dout,din)-1)*(np.sqrt(6)/np.sqrt(din*dout))