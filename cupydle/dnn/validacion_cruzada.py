#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__      = "Ponzoni, Nelson"
__copyright__   = "Copyright 2015"
__credits__     = ["Ponzoni Nelson"]
__maintainer__  = "Ponzoni Nelson"
__contact__     = "npcuadra@gmail.com"
__email__       = "npcuadra@gmail.com"
__license__     = "GPL"
__version__     = "1.0.0"
__status__      = "Production"

"""
El :mod:`cupydle.validacion_cruzada` uncluye las funciones de ayuda para
implementar tecnicas de validacion cruzada sobre los datos.
"""

import numpy as np
from numpy import bincount
import warnings
from abc import ABCMeta, abstractmethod
from itertools import combinations
from math import ceil, floor, factorial


# solo permito estos imports
__all__ = ['KFold',
           'LabelKFold',
           'LeaveOneOut',
           'StratifiedKFold']



class _PartitionIterator(metaclass=ABCMeta):
    """Base class for CV iterators where train_mask = ~test_mask
    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    Parameters
    ----------
    n : int
        Total number of elements in dataset.
    """

    def __init__(self, n):
        if abs(n - int(n)) >= np.finfo('f').eps:
            raise ValueError("n must be an integer")
        self.n = int(n)

    def __iter__(self):
        ind = np.arange(self.n)
        for test_index in self._iter_test_masks():
            train_index = np.logical_not(test_index)
            train_index = ind[train_index]
            test_index = ind[test_index]
            yield train_index, test_index

    # Since subclasses must implement either _iter_test_masks or
    # _iter_test_indices, neither can be abstract.
    def _iter_test_masks(self):
        """Generates boolean masks corresponding to test sets.
        By default, delegates to _iter_test_indices()
        """
        for test_index in self._iter_test_indices():
            test_mask = self._empty_mask()
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self):
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError

    def _empty_mask(self):
        return np.zeros(self.n, dtype=np.bool)

class LeaveOneOut(_PartitionIterator):
    """Leave-One-Out cross validation iterator.
    Provides train/test indices to split data in train test sets. Each
    sample is used once as a test set (singleton) while the remaining
    samples form the training set.
    Note: ``LeaveOneOut(n)`` is equivalent to ``KFold(n, n_folds=n)`` and
    ``LeavePOut(n, p=1)``.
    Due to the high number of test sets (which is the same as the
    number of samples) this cross validation method can be very costly.
    For large datasets one should favor KFold, StratifiedKFold or
    ShuffleSplit.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n : int
        Total number of elements in dataset.
    Examples
    --------
    >>> from sklearn import cross_validation
    >>> X = np.array([[1, 2], [3, 4]])
    >>> y = np.array([1, 2])
    >>> loo = cross_validation.LeaveOneOut(2)
    >>> len(loo)
    2
    >>> print(loo)
    sklearn.cross_validation.LeaveOneOut(n=2)
    >>> for train_index, test_index in loo:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    ...    print(X_train, X_test, y_train, y_test)
    TRAIN: [1] TEST: [0]
    [[3 4]] [[1 2]] [2] [1]
    TRAIN: [0] TEST: [1]
    [[1 2]] [[3 4]] [1] [2]
    See also
    --------
    LeaveOneLabelOut for splitting the data according to explicit,
    domain-specific stratification of the dataset.
    """

    def _iter_test_indices(self):
        return range(self.n)

    def __repr__(self):
        return '%s.%s(n=%i)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
        )

    def __len__(self):
        return self.n

class _BaseKFold(_PartitionIterator):
    """Base class to validate KFold approaches"""

    @abstractmethod
    def __init__(self, n, n_folds, shuffle, random_state):
        super(_BaseKFold, self).__init__(n)

        if abs(n_folds - int(n_folds)) >= np.finfo('f').eps:
            raise ValueError("n_folds must be an integer")
        self.n_folds = n_folds = int(n_folds)

        if n_folds <= 1:
            raise ValueError(
                "k-fold cross validation requires at least one"
                " train / test split by setting n_folds=2 or more,"
                " got n_folds={0}.".format(n_folds))
        if n_folds > self.n:
            raise ValueError(
                ("Cannot have number of folds n_folds={0} greater"
                 " than the number of samples: {1}.").format(n_folds, n))

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False;"
                            " got {0}".format(shuffle))
        self.shuffle = shuffle
        self.random_state = random_state

class KFold(_BaseKFold):
    """K-Folds cross validation iterator.
    Provides train/test indices to split data in train test sets. Split
    dataset into k consecutive folds (without shuffling by default).
    Each fold is then used a validation set once while the k - 1 remaining
    fold form the training set.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n : int
        Total number of elements.
    n_folds : int, default=3
        Number of folds. Must be at least 2.
    shuffle : boolean, optional
        Whether to shuffle the data before splitting into batches.
    random_state : None, int or RandomState
        When shuffle=True, pseudo-random number generator state used for
        shuffling. If None, use default numpy RNG for shuffling.
    Examples
    --------
    >>> from sklearn.cross_validation import KFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4])
    >>> kf = KFold(4, n_folds=2)
    >>> len(kf)
    2
    >>> print(kf)  # doctest: +NORMALIZE_WHITESPACE
    sklearn.cross_validation.KFold(n=4, n_folds=2, shuffle=False,
                                   random_state=None)
    >>> for train_index, test_index in kf:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [2 3] TEST: [0 1]
    TRAIN: [0 1] TEST: [2 3]
    Notes
    -----
    The first n % n_folds folds have size n // n_folds + 1, other folds have
    size n // n_folds.
    See also
    --------
    StratifiedKFold: take label information into account to avoid building
    folds with imbalanced class distributions (for binary or multiclass
    classification tasks).
    LabelKFold: K-fold iterator variant with non-overlapping labels.
    """

    def __init__(self, n, n_folds=3, shuffle=False,
                 random_state=None):
        super(KFold, self).__init__(n, n_folds, shuffle, random_state)
        self.idxs = np.arange(n)
        if shuffle:
            rng = check_random_state(self.random_state)
            rng.shuffle(self.idxs)

    def _iter_test_indices(self):
        n = self.n
        n_folds = self.n_folds
        fold_sizes = (n // n_folds) * np.ones(n_folds, dtype=np.int)
        fold_sizes[:n % n_folds] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield self.idxs[start:stop]
            current = stop

    def __repr__(self):
        return '%s.%s(n=%i, n_folds=%i, shuffle=%s, random_state=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
            self.n_folds,
            self.shuffle,
            self.random_state,
        )

    def __len__(self):
        return self.n_folds

class LabelKFold(_BaseKFold):
    """K-fold iterator variant with non-overlapping labels.
    The same label will not appear in two different folds (the number of
    distinct labels has to be at least equal to the number of folds).
    The folds are approximately balanced in the sense that the number of
    distinct labels is approximately the same in each fold.
    .. versionadded:: 0.17
    Parameters
    ----------
    labels : array-like with shape (n_samples, )
        Contains a label for each sample.
        The folds are built so that the same label does not appear in two
        different folds.
    n_folds : int, default=3
        Number of folds. Must be at least 2.
    Examples
    --------
    >>> from sklearn.cross_validation import LabelKFold
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1, 2, 3, 4])
    >>> labels = np.array([0, 0, 2, 2])
    >>> label_kfold = LabelKFold(labels, n_folds=2)
    >>> len(label_kfold)
    2
    >>> print(label_kfold)
    sklearn.cross_validation.LabelKFold(n_labels=4, n_folds=2)
    >>> for train_index, test_index in label_kfold:
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    ...     print(X_train, X_test, y_train, y_test)
    ...
    TRAIN: [0 1] TEST: [2 3]
    [[1 2]
     [3 4]] [[5 6]
     [7 8]] [1 2] [3 4]
    TRAIN: [2 3] TEST: [0 1]
    [[5 6]
     [7 8]] [[1 2]
     [3 4]] [3 4] [1 2]
    See also
    --------
    LeaveOneLabelOut for splitting the data according to explicit,
    domain-specific stratification of the dataset.
    """
    def __init__(self, labels, n_folds=3):
        super(LabelKFold, self).__init__(len(labels), n_folds,
                                         shuffle=False, random_state=None)

        unique_labels, labels = np.unique(labels, return_inverse=True)
        n_labels = len(unique_labels)

        if n_folds > n_labels:
            raise ValueError(
                    ("Cannot have number of folds n_folds={0} greater"
                     " than the number of labels: {1}.").format(n_folds,
                                                                n_labels))

        # Weight labels by their number of occurences
        n_samples_per_label = np.bincount(labels)

        # Distribute the most frequent labels first
        indices = np.argsort(n_samples_per_label)[::-1]
        n_samples_per_label = n_samples_per_label[indices]

        # Total weight of each fold
        n_samples_per_fold = np.zeros(n_folds)

        # Mapping from label index to fold index
        label_to_fold = np.zeros(len(unique_labels))

        # Distribute samples by adding the largest weight to the lightest fold
        for label_index, weight in enumerate(n_samples_per_label):
            lightest_fold = np.argmin(n_samples_per_fold)
            n_samples_per_fold[lightest_fold] += weight
            label_to_fold[indices[label_index]] = lightest_fold

        self.idxs = label_to_fold[labels]

    def _iter_test_indices(self):
        for f in range(self.n_folds):
            yield np.where(self.idxs == f)[0]

    def __repr__(self):
        return '{0}.{1}(n_labels={2}, n_folds={3})'.format(
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
            self.n_folds,
        )

    def __len__(self):
        return self.n_folds

class StratifiedKFold(_BaseKFold):
    """Stratified K-Folds cross validation iterator
    Provides train/test indices to split data in train test sets.
    This cross-validation object is a variation of KFold that
    returns stratified folds. The folds are made by preserving
    the percentage of samples for each class.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    y : array-like, [n_samples]
        Samples to split in K folds.
    n_folds : int, default=3
        Number of folds. Must be at least 2.
    shuffle : boolean, optional
        Whether to shuffle each stratification of the data before splitting
        into batches.
    random_state : None, int or RandomState
        When shuffle=True, pseudo-random number generator state used for
        shuffling. If None, use default numpy RNG for shuffling.
    Examples
    --------
    >>> from sklearn.cross_validation import StratifiedKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> skf = StratifiedKFold(y, n_folds=2)
    >>> len(skf)
    2
    >>> print(skf)  # doctest: +NORMALIZE_WHITESPACE
    sklearn.cross_validation.StratifiedKFold(labels=[0 0 1 1], n_folds=2,
                                             shuffle=False, random_state=None)
    >>> for train_index, test_index in skf:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [1 3] TEST: [0 2]
    TRAIN: [0 2] TEST: [1 3]
    Notes
    -----
    All the folds have size trunc(n_samples / n_folds), the last one has the
    complementary.
    See also
    --------
    LabelKFold: K-fold iterator variant with non-overlapping labels.
    """

    def __init__(self, y, n_folds=3, shuffle=False,
                 random_state=None):
        super(StratifiedKFold, self).__init__(
            len(y), n_folds, shuffle, random_state)
        y = np.asarray(y)
        n_samples = y.shape[0]
        unique_labels, y_inversed = np.unique(y, return_inverse=True)
        label_counts = bincount(y_inversed)
        min_labels = np.min(label_counts)
        if self.n_folds > min_labels:
            warnings.warn(("The least populated class in y has only %d"
                           " members, which is too few. The minimum"
                           " number of labels for any class cannot"
                           " be less than n_folds=%d."
                           % (min_labels, self.n_folds)), Warning)

        # don't want to use the same seed in each label's shuffle
        if self.shuffle:
            rng = check_random_state(self.random_state)
        else:
            rng = self.random_state

        # pre-assign each sample to a test fold index using individual KFold
        # splitting strategies for each label so as to respect the
        # balance of labels
        per_label_cvs = [
            KFold(max(c, self.n_folds), self.n_folds, shuffle=self.shuffle,
                  random_state=rng) for c in label_counts]
        test_folds = np.zeros(n_samples, dtype=np.int)
        for test_fold_idx, per_label_splits in enumerate(zip(*per_label_cvs)):
            for label, (_, test_split) in zip(unique_labels, per_label_splits):
                label_test_folds = test_folds[y == label]
                # the test split can be too big because we used
                # KFold(max(c, self.n_folds), self.n_folds) instead of
                # KFold(c, self.n_folds) to make it possible to not crash even
                # if the data is not 100% stratifiable for all the labels
                # (we use a warning instead of raising an exception)
                # If this is the case, let's trim it:
                test_split = test_split[test_split < len(label_test_folds)]
                label_test_folds[test_split] = test_fold_idx
                test_folds[y == label] = label_test_folds

        self.test_folds = test_folds
        self.y = y

    def _iter_test_masks(self):
        for i in range(self.n_folds):
            yield self.test_folds == i

    def __repr__(self):
        return '%s.%s(labels=%s, n_folds=%i, shuffle=%s, random_state=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.y,
            self.n_folds,
            self.shuffle,
            self.random_state,
        )

    def __len__(self):
        return self.n_folds

if __name__ == '__main__':
    """
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [5,6], [7,8], [1, 2], [3, 4], [5,6], [7,8]])
    y = np.array([0, 0, 1, 1, 1,0,3,3,3,0])
    print(X)
    print(y)
    skf = StratifiedKFold(y, n_folds=3)
    print(len(skf))
    print(skf)  # doctest: +NORMALIZE_WHITESPACE
    #cross_validation.StratifiedKFold(labels=[0 0 1 1], n_folds=2, shuffle=False, random_state=None)
    for train_index, test_index in skf:
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    """
    assert False, str(__file__ + " No es un modulo")
