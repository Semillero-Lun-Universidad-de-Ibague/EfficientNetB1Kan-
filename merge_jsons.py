import json

# Seznam všech JSON souborů, které chceš sloučit.
json_files = ["data.json", "combined.json"]

# Prázdný slovník pro uložení sloučených dat.
combined_data = {}

# Načti každý JSON soubor a aktualizuj slovník.
for json_file in json_files:
    with open(json_file, "r") as f:
        data = json.load(f)
        combined_data.update(data)

# Ulož sloučená data do jednoho JSON souboru.
with open("combined.json", "w") as f:
    json.dump(combined_data, f, indent=4)
