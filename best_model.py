import json
from typing import List, Dict
import numpy as np
from scipy import stats


def load_models(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_learning_slope(accuracies: List[float]) -> float:
    epochs = np.arange(len(accuracies))
    slope, _, _, _, _ = stats.linregress(epochs, accuracies)
    return slope


def sort_models_by_learning_speed(models: List[Dict]) -> List[Dict]:
    # Výpočet směrnice pro každý model a vytvoření seznamu dvojic (model, směrnice)
    model_slopes = []
    for model in models:
        if 'accuracy_validation' not in model or not model['accuracy_validation']:
            print(f"Varování: Model '{model.get('model_name', 'Neznámý')}' nemá data accuracy_validation. Přeskakuji.")
            continue
        slope = calculate_learning_slope(model['accuracy_validation'])
        model_slopes.append((model, slope))

    # Seřazení seznamu podle směrnice (sestupně)
    sorted_models = sorted(model_slopes, key=lambda x: x[1], reverse=True)

    # Vrácení pouze seřazených modelů (bez směrnic)
    return [model for model, _ in sorted_models]


# Použití funkcí
file_path = 'dalsi.json'  # Nahraďte skutečnou cestou k vašemu souboru
models = load_models(file_path)
sorted_models = sort_models_by_learning_speed(models)

# Výpis seřazených modelů
for i, model in enumerate(sorted_models, 1):
    slope = calculate_learning_slope(model['accuracy_validation'])
    print(f"{i}. Model: {model['model_name']}, Směrnice učení: {slope:.6f}")