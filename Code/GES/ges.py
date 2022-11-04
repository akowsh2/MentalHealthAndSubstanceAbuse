from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io
import numpy as np

data = np.loadtxt('../Data/processedData.txt', skiprows=1)

score_functions = ["local_score_BIC", "local_score_BDeu"]
labels = ["Age", "EmploymentStatus", "IncomeLevel" , "Urbal/Rural", "DrugUse", "MentalIllness"]

    # Causal Discovery
for score_func in score_functions:
    Record = ges(data, score_func)
    print("Causal discovery complete")

    # Visualization
    pyd = GraphUtils.to_pydot(Record['G'], labels=labels)
    tmp_png = pyd.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')

    # or save the graph
    pyd.write_png(score_func + '.png')

print("Finished execution")