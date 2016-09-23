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

from itertools import chain


from cupydle.dnn.utils import safe_indexing, indexable, check_consistent_length, check_random_state


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






class BaseShuffleSplit(metaclass=ABCMeta):
    """Base class for ShuffleSplit and StratifiedShuffleSplit"""

    def __init__(self, n, n_iter=10, test_size=0.1, train_size=None,
                 random_state=None):
        self.n = n
        self.n_iter = n_iter
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.n_train, self.n_test = _validate_shuffle_split(n, test_size,
                                                            train_size)

    def __iter__(self):
        for train, test in self._iter_indices():
            yield train, test
        return

    @abstractmethod
    def _iter_indices(self):
        """Generate (train, test) indices"""


class ShuffleSplit(BaseShuffleSplit):
    """Random permutation cross-validation iterator.
    Yields indices to split data into training and test sets.
    Note: contrary to other cross-validation strategies, random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n : int
        Total number of elements in the dataset.
    n_iter : int (default 10)
        Number of re-shuffling & splitting iterations.
    test_size : float (default 0.1), int, or None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.
    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.
    Examples
    --------
    >>> from sklearn import cross_validation
    >>> rs = cross_validation.ShuffleSplit(4, n_iter=3,
    ...     test_size=.25, random_state=0)
    >>> len(rs)
    3
    >>> print(rs)
    ... # doctest: +ELLIPSIS
    ShuffleSplit(4, n_iter=3, test_size=0.25, ...)
    >>> for train_index, test_index in rs:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...
    TRAIN: [3 1 0] TEST: [2]
    TRAIN: [2 1 3] TEST: [0]
    TRAIN: [0 2 1] TEST: [3]
    >>> rs = cross_validation.ShuffleSplit(4, n_iter=3,
    ...     train_size=0.5, test_size=.25, random_state=0)
    >>> for train_index, test_index in rs:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...
    TRAIN: [3 1] TEST: [2]
    TRAIN: [2 1] TEST: [0]
    TRAIN: [0 2] TEST: [3]
    """

    def _iter_indices(self):
        rng = check_random_state(self.random_state)
        for i in range(self.n_iter):
            # random partition
            permutation = rng.permutation(self.n)
            ind_test = permutation[:self.n_test]
            ind_train = permutation[self.n_test:self.n_test + self.n_train]
            yield ind_train, ind_test

    def __repr__(self):
        return ('%s(%d, n_iter=%d, test_size=%s, '
                'random_state=%s)' % (
                    self.__class__.__name__,
                    self.n,
                    self.n_iter,
                    str(self.test_size),
                    self.random_state,
                ))

    def __len__(self):
        return self.n_iter


def _validate_shuffle_split(n, test_size, train_size):
    if test_size is None and train_size is None:
        raise ValueError(
            'test_size and train_size can not both be None')

    if test_size is not None:
        if np.asarray(test_size).dtype.kind == 'f':
            if test_size >= 1.:
                raise ValueError(
                    'test_size=%f should be smaller '
                    'than 1.0 or be an integer' % test_size)
        elif np.asarray(test_size).dtype.kind == 'i':
            if test_size >= n:
                raise ValueError(
                    'test_size=%d should be smaller '
                    'than the number of samples %d' % (test_size, n))
        else:
            raise ValueError("Invalid value for test_size: %r" % test_size)

    if train_size is not None:
        if np.asarray(train_size).dtype.kind == 'f':
            if train_size >= 1.:
                raise ValueError("train_size=%f should be smaller "
                                 "than 1.0 or be an integer" % train_size)
            elif np.asarray(test_size).dtype.kind == 'f' and \
                    train_size + test_size > 1.:
                raise ValueError('The sum of test_size and train_size = %f, '
                                 'should be smaller than 1.0. Reduce '
                                 'test_size and/or train_size.' %
                                 (train_size + test_size))
        elif np.asarray(train_size).dtype.kind == 'i':
            if train_size >= n:
                raise ValueError("train_size=%d should be smaller "
                                 "than the number of samples %d" %
                                 (train_size, n))
        else:
            raise ValueError("Invalid value for train_size: %r" % train_size)

    if np.asarray(test_size).dtype.kind == 'f':
        n_test = ceil(test_size * n)
    elif np.asarray(test_size).dtype.kind == 'i':
        n_test = float(test_size)

    if train_size is None:
        n_train = n - n_test
    else:
        if np.asarray(train_size).dtype.kind == 'f':
            n_train = floor(train_size * n)
        else:
            n_train = float(train_size)

    if test_size is None:
        n_test = n - n_train

    if n_train + n_test > n:
        raise ValueError('The sum of train_size and test_size = %d, '
                         'should be smaller than the number of '
                         'samples %d. Reduce test_size and/or '
                         'train_size.' % (n_train + n_test, n))

    return int(n_train), int(n_test)


class StratifiedShuffleSplit(BaseShuffleSplit):
    """Stratified ShuffleSplit cross validation iterator
    Provides train/test indices to split data in train test sets.
    This cross-validation object is a merge of StratifiedKFold and
    ShuffleSplit, which returns stratified randomized folds. The folds
    are made by preserving the percentage of samples for each class.
    Note: like the ShuffleSplit strategy, stratified random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    y : array, [n_samples]
        Labels of samples.
    n_iter : int (default 10)
        Number of re-shuffling & splitting iterations.
    test_size : float (default 0.1), int, or None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.
    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.
    Examples
    --------
    >>> from sklearn.cross_validation import StratifiedShuffleSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> sss = StratifiedShuffleSplit(y, 3, test_size=0.5, random_state=0)
    >>> len(sss)
    3
    >>> print(sss)       # doctest: +ELLIPSIS
    StratifiedShuffleSplit(labels=[0 0 1 1], n_iter=3, ...)
    >>> for train_index, test_index in sss:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [1 2] TEST: [3 0]
    TRAIN: [0 2] TEST: [1 3]
    TRAIN: [0 2] TEST: [3 1]
    """

    def __init__(self, y, n_iter=10, test_size=0.1, train_size=None,
                 random_state=None):

        super(StratifiedShuffleSplit, self).__init__(
            len(y), n_iter, test_size, train_size, random_state)

        self.y = np.array(y)
        self.classes, self.y_indices = np.unique(y, return_inverse=True)
        n_cls = self.classes.shape[0]

        if np.min(bincount(self.y_indices)) < 2:
            raise ValueError("The least populated class in y has only 1"
                             " member, which is too few. The minimum"
                             " number of labels for any class cannot"
                             " be less than 2.")

        if self.n_train < n_cls:
            raise ValueError('The train_size = %d should be greater or '
                             'equal to the number of classes = %d' %
                             (self.n_train, n_cls))
        if self.n_test < n_cls:
            raise ValueError('The test_size = %d should be greater or '
                             'equal to the number of classes = %d' %
                             (self.n_test, n_cls))

    def _iter_indices(self):
        rng = check_random_state(self.random_state)
        cls_count = bincount(self.y_indices)
        p_i = cls_count / float(self.n)
        n_i = np.round(self.n_train * p_i).astype(int)
        t_i = np.minimum(cls_count - n_i,
                         np.round(self.n_test * p_i).astype(int))

        for n in range(self.n_iter):
            train = []
            test = []

            for i, cls in enumerate(self.classes):
                permutation = rng.permutation(cls_count[i])
                cls_i = np.where((self.y == cls))[0][permutation]

                train.extend(cls_i[:n_i[i]])
                test.extend(cls_i[n_i[i]:n_i[i] + t_i[i]])

            # Because of rounding issues (as n_train and n_test are not
            # dividers of the number of elements per class), we may end
            # up here with less samples in train and test than asked for.
            if len(train) < self.n_train or len(test) < self.n_test:
                # We complete by affecting randomly the missing indexes
                missing_idx = np.where(bincount(train + test,
                                                minlength=len(self.y)) == 0,
                                       )[0]
                missing_idx = rng.permutation(missing_idx)
                train.extend(missing_idx[:(self.n_train - len(train))])
                test.extend(missing_idx[-(self.n_test - len(test)):])

            train = rng.permutation(train)
            test = rng.permutation(test)

            yield train, test

    def __repr__(self):
        return ('%s(labels=%s, n_iter=%d, test_size=%s, '
                'random_state=%s)' % (
                    self.__class__.__name__,
                    self.y,
                    self.n_iter,
                    str(self.test_size),
                    self.random_state,
                ))

    def __len__(self):
        return self.n_iter


def _num_samples(x):
    """Return number of samples in array-like x."""
    if hasattr(x, 'fit'):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)

def train_test_split(*arrays, **options):
    """Split arrays or matrices into random train and test subsets
    Quick utility that wraps input validation and
    ``next(iter(ShuffleSplit(n_samples)))`` and application to input
    data into a single call for splitting (and optionally subsampling)
    data in a oneliner.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    test_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.
        If train size is also None, test size is set to 0.25.
    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.
    stratify : array-like or None (default is None)
        If not None, data is split in a stratified fashion, using this as
        Consiste en la división previa de la población de estudio en grupos o
        clases que se suponen homogéneos respecto a característica a estudiar y que no se solapen.
        the labels array.
    Returns
    -------
    splitting : list, length = 2 * len(arrays),
        List containing train-test split of inputs.
        .. versionadded:: 0.16
            Output type is the same as the input type.
    Examples
    --------
    >>> import numpy as np
    >>> from cupydle.dnn.validacion_cruzada import train_test_split
    >>> X, y = np.arange(10).reshape((5, 2)), range(5)
    >>> X
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])
    >>> list(y)
    [0, 1, 2, 3, 4]
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.33, random_state=42)
    ...
    >>> X_train
    array([[4, 5],
           [0, 1],
           [6, 7]])
    >>> y_train
    [2, 0, 3]
    >>> X_test
    array([[2, 3],
           [8, 9]])
    >>> y_test
    [1, 4]
    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    test_size = options.pop('test_size', None)
    train_size = options.pop('train_size', None)
    random_state = options.pop('random_state', None)
    dtype = options.pop('dtype', None)
    if dtype is not None:
        warnings.warn("dtype option is ignored and will be removed in 0.18.",
                      DeprecationWarning)

    allow_nd = options.pop('allow_nd', None)
    allow_lists = options.pop('allow_lists', None)
    stratify = options.pop('stratify', None)

    if allow_lists is not None:
        warnings.warn("The allow_lists option is deprecated and will be "
                      "assumed True in 0.18 and removed.", DeprecationWarning)

    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))
    if allow_nd is not None:
        warnings.warn("The allow_nd option is deprecated and will be "
                      "assumed True in 0.18 and removed.", DeprecationWarning)
    if allow_lists is False or allow_nd is False:
        arrays = [check_array(x, 'csr', allow_nd=allow_nd,
                              force_all_finite=False, ensure_2d=False)
                  if x is not None else x
                  for x in arrays]

    if test_size is None and train_size is None:
        test_size = 0.25
#    arrays = indexable(*arrays)
    if stratify is not None:
        cv = StratifiedShuffleSplit(stratify, test_size=test_size,
                                    train_size=train_size,
                                    random_state=random_state)
    else:
        n_samples = _num_samples(arrays[0])
        cv = ShuffleSplit(n_samples, test_size=test_size,
                          train_size=train_size,
                          random_state=random_state)

    train, test = next(iter(cv))
    return list(chain.from_iterable((safe_indexing(a, train),
                                     safe_indexing(a, test)) for a in arrays))

if __name__ == '__main__':

    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [5,6], [7,8], [1, 2], [3, 4], [5,6], [7,8]])
    y = np.array([0, 0, 1, 1, 1,0,3,3,3,0])
    print(X)
    print(y)
    skf = StratifiedKFold(y, n_folds=2)
    print(len(skf))
    print(skf)  # doctest: +NORMALIZE_WHITESPACE
    #cross_validation.StratifiedKFold(labels=[0 0 1 1], n_folds=2, shuffle=False, random_state=None)
    for train_index, test_index in skf:
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    print()
    import numpy as np
    X, y = np.arange(10).reshape((5, 2)), range(5)
    print(X)
    print(list(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)


    assert False, str(__file__ + " No es un modulo")
