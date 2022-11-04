from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
import numpy as np

data = np.loadtxt('../Data/processedData.txt', skiprows=1)

independece_test_methods = ["fisherz", "chisq", "gsq", "mv_fisherz"]
labels = ["Age", "EmploymentStatus", "IncomeLevel" , "Urbal/Rural", "DrugUse", "MentalIllness"]

    # Causal Discovery
for test_method in independece_test_methods:
    G, edges = fci(data, test_method, verbose=True)
    print("Causal discovery complete")

    # Visualization
    pdy = GraphUtils.to_pydot(G, labels=labels)
    pdy.write_png(test_method + ".png")

print("Finished execution")