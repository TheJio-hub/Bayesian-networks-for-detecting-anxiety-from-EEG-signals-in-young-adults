from pgmpy.datasets import load_dataset
from pgmpy.causal_discovery import PC

dataset = load_dataset("sachs_discrete")
est = PC(ci_test="chi_square", return_type="dag")
est.fit(dataset.data)
print(est.causal_graph_)