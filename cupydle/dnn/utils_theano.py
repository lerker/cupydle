
__author__      = "Ponzoni, Nelson"
__copyright__   = "Copyright 2015"
__credits__     = ["Ponzoni Nelson"]
__maintainer__  = "Ponzoni Nelson"
__contact__     = "npcuadra@gmail.com"
__email__       = "npcuadra@gmail.com"
__license__     = "GPL"
__version__     = "1.0.0"
__status__      = "Production"


import theano


# colores para los grafos, puedo cambiar lo que quiera del diccionario y pasarlo
default_colorCodes = {'GpuFromHost': 'red',
                      'HostFromGpu': 'red',
                      'Scan': 'yellow',
                      'Shape': 'brown',
                      'IfElse': 'magenta',
                      'Elemwise': '#FFAABB',  # dark pink
                      'Subtensor': '#FFAAFF',  # purple
                      'Alloc': '#FFAA22',  # orange
                      'Output': 'lightblue'}


def plot_graph(graph, name=None, path=None):
    """
    plot a graph of theano object (function, nodes etc)
    """

    if name is None:
        name = "symbolic_graph_operation.pdf"
    if path is None:
        path = "cupydle/test/"

    filepath = path + name

    theano.printing.pydotprint( fct=graph,
                                outfile=filepath,
                                format='pdf',
                                compact=True, # no imprime variables sin nombre
                                with_ids=True, # numero de nodos
                                high_contrast=True,
                                scan_graphs=True, # imprime los scans
                                cond_highlight=True,
                                var_with_name_simple=True, #si la variable tiene nombre, solo imprime eso
                                colorCodes=default_colorCodes) # codigo de colores
    return 1