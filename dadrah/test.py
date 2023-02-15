import lmfit

from lmfit.models import PolynomialModel
from lmfit.model import save_model, save_modelresult
    
model = PolynomialModel(degree=15)

