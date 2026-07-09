import faiss
import numpy as np
from indexes.base import BaseIndex

class HNSWIndex(BaseIndex):
    #ANN search will have one param ef to know when to stop search (for constructor)