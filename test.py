from services.hybrid_predictor import hybrid_predict
import sys
sys.stdout.reconfigure(encoding='utf-8')

result = hybrid_predict("data/uploads/sample_leaf.JPG")
print(result["stage"])
print(result["message"])
