__author__ = 'lerker'

# Dependencias externas
#from scipy.io import loadmat, savemat
import numpy as np

text_extensions = ['.dat', '.txt', '.csv']


def parse_point(line):
    # TODO dar posibilidad de cambiar separador
    values = [float(x) for x in line.split(';')]
    return values[-1], values[0:-1]


# Checks
def is_text_file(path):
    return any([path.lower().endswith(ext) for ext in text_extensions])


def is_mat_file(path):
    return path.lower().endswith('.mat')


# Loader TODO mejorar manejo de .mat
def load_file(path, separador=";", varname=None):
    if is_text_file(path):
        dataset = np.loadtxt(path, delimiter=separador)
    elif is_mat_file(path):
        #dataset = loadmat(path)[varname]
    else:
        assert(True, "Tipo de archivo no reconocido")
    return dataset

# Saver TODO mejorar esto que ni anda
def save_file(data, path):
    if is_text_file(path):
        data.saveAsTextFile(path+'.txt')
    elif is_mat_file(path):
        #savemat(path, {'dataset': data.collect()})
    else:
        data.saveAsPickleFile(path)
    return


def load_iris():
    """Load and return the iris dataset (classification).
    The iris dataset is a classic and very easy multi-class classification
    dataset.
    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============
    Read more in the :ref:`User Guide <datasets>`.
    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels,
        'target_names', the meaning of the labels, 'feature_names', the
        meaning of the features, and 'DESCR', the
        full description of the dataset.
    Examples
    --------
    Let's say you are interested in the samples 10, 25, and 50, and want to
    know their class name.
    -> from sklearn.datasets import load_iris
    -> data = load_iris()
    -> data.target[[10, 25, 50]]
    array([0, 0, 1])
    -> list(data.target_names)
    ['setosa', 'versicolor', 'virginica']
    """
    module_path = dirname(__file__)
    with open(join(module_path, 'data', 'iris.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=np.int)

    with open(join(module_path, 'descr', 'iris.rst')) as rst_file:
        fdescr = rst_file.read()

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['sepal length (cm)', 'sepal width (cm)',
                                'petal length (cm)', 'petal width (cm)'])

