from models import evaluate_model_kfold
from datas import DATASET


result = evaluate_model_kfold(only_rvsm=True)
print(result)
with open(DATASET.results / 'results.txt', 'a') as f:
    f.write(str(result))
    f.write('\n')