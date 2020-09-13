# Copyright 2020 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array
from scipy.stats import multiscale_graphcorr

from hyppo.independence import Dcorr, RV, CCA


class BaseSubgraph(BaseEstimator):
    """
    A base class for estimating the signal subgraph (ss).

    Parameters
    ----------
    alg : string
        Algorithm to use to estimate signal subgraph. Options are
        "screen", "itscreen", "coherence", or "sparse". Defaulted to "screen".
    stat : string
        Desired test statistic to use on data if alg is "screen". If "mgc",
        mulstiscale graph correlation will be used. Otherwise, must be
        either "dcorr", "rv", or "cca". Defaulted to "mgc".
    constraints: int or vector
        The constraints that will be imposed onto the estimated ss.

        If constraints is an int, constraints is the number of edges
        in the ss.

        If constraints is a vector, then its first element is the number
        of edges in the ss, and its second element is the number of
        vertices that the ss must be incident to.

    See Also
    --------
    graspy.subgraph.screen, graspy.subgraph.itscreen, graspy.subgraph.coherence, graspy.subgraph.parse
    """

    def __init__(
        self,
        stat=None,
    ):
        stats = [None, "mgc", "dcorr", "rv", "cca"]

        if stat not in stats:
            msg = 'stat must be either "mgc", "dcorr", "rv", or "cca".'
            raise ValueError(msg)
        else:
            self.stat = stat

    def _screen(self, X, y):
        """
        Performs non-iterative screening on graphs to estimate signal subgraph.

        Parameters
        ----------
        X: np.ndarray, shape (n_graphs, n_vertices, n_vertices)
            Tensor of adjacency matrices.
        y: np.ndarray, shape (n_graphs, 1)
            Vector of ground truth labels.

        Attributes
        ----------
        n_graphs: int
            Number of graphs in X
        n_vertices: int
            Dimension of each graph in X

        Returns
        -------
        corrs: np.ndarray, shape (n_vertices, 1)
            Vector of correlation values for each node.

        References
        ----------
        .. [1] S. Wang, C. Chen, A. Badea, Priebe, C.E., Vogelstein, J.T.
        "Signal Subgraph Estimation Via Vertex Screening" arXiv: 1801.07683
        [stat.ME], 2018
        """

        # Error Checking
        if type(X) is not np.ndarray:
            raise TypeError("X must be numpy.ndarray")
        if type(y) is not np.ndarray:
            raise TypeError("y must be numpy.ndarray")

        check_array(X, dtype=int, ensure_2d=False, allow_nd=True)
        check_array(y, dtype=int)

        # Finding dimensions
        self.n_graphs = X.shape[0]
        self.n_vertices = X.shape[-1]

        if len(X.shape) != 3:
            raise ValueError("X must be a tensor")
        if X.shape[1] != X.shape[2]:
            raise ValueError("Entries in X must be square matrices")

        if y.shape != (self.n_graphs, 1):
            raise ValueError("y must have shape (n_graphs, 1)")

        corrs = np.zeros((self.n_vertices, 1))
        for i in range(self.n_vertices):

            # Stacks the ith row of each matrix in tensor,
            # creates matrix with dimension n_graphs by n_vertices
            mat = X[:, i]

            # Statistical measurement chosen by the user
            if self.stat == "mgc":
                c_u, p_value, mgc_dict = multiscale_graphcorr(mat, y, reps=1)

            else:
                if self.stat == "dcorr":
                    test = Dcorr()
                elif self.stat == "rv":
                    test = RV()
                else:
                    test = CCA()
                c_u = test._statistic(mat, y)

            corrs[i][0] = c_u

        return corrs

    @abstractmethod
    def fit(self, X, y):
        """
        A method for signal subgraph estimation.

        Parameters
        ----------
        X: np.ndarray
        y: np.ndarray

        Returns
        -------
        self : returns an instance of self.
        """

        return self


class BaseClassify(ABC, BaseEstimator, ClassifierMixin):
    """
    Base classification class.

    See Also
    --------
    graspy.subgraph.sparse_opt
    """
