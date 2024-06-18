from calcul_with_GEN import all_calcul_GEN as acgen
from view_result import all_view
from graph_for_rank_in_lora import make_graph

basedesc = "mt0_hate_finedtune"
list_to_test = ['r2', 'r4', 'r8', 'r16', 'r16-1', 'r16-2', 'r16-3', 'r16-4', 'r32']
case = 'hatecheck_base'

for r in list_to_test:
	
	# make prediction for {basedesc}_{r}
	acgen('data', case=case, desc=f"{basedesc}_{r}")

	# make graph for {basedesc}_{r}
	all_view('data', desc=f"{basedesc}_{r}", onlyOne=case)


# Make rank plots
make_graph(basedesc, list_to_test, case=case)

#Towards an Undiased Compression of Language Models