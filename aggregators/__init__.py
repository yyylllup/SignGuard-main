
from .Mean import *
from .Median import *
from .GeoMed import *
from .Krum import *
from .Bulyan import *
from .signguard import *
from .DnC import *

def aggregator(rule):
    # gradient aggregation rule
    GAR = {'Mean':mean,
           'TrMean':trimmed_mean,
           'Median':median,
           'GeoMed':geomed,
           'Multi-Krum':multi_krum,
           'Bulyan':bulyan,
           'DnC':divide_conquer,
           'SignGuard': signguard_multiclass,
           'SignGuard-Sim': signguard_multiclass_plus1,
           'SignGuard-Dist': signguard_multiclass_plus2,
    }

    return GAR[rule]