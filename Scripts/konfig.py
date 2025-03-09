# Funkcja Python do automatycznego tworzenia pliku konfiguracyjnego data.yaml
# 1. Odczytuje plik "classes.txt", aby uzyskać listę nazw klas
# 2. Tworzy słownik danych z poprawnymi ścieżkami do folderów, liczbą klas i nazwami klas
# 3. Zapisuje dane w formacie YAML do pliku data.yaml

import yaml
import os

def create_data_yaml(path_to_classes_txt, path_to_data_yaml):
    # Sprawdzenie, czy plik classes.txt istnieje
    if not os.path.exists(path_to_classes_txt):
        print(f'classes.txt file not found! Please create a classes.txt labelmap and move it to {path_to_classes_txt}')
        return
    
    # Odczytanie nazw klas z pliku classes.txt
    with open(path_to_classes_txt, 'r') as f:
        classes = []
        for line in f.readlines():
            if len(line.strip()) == 0:  # Pomijanie pustych linii
                continue
            classes.append(line.strip())
    number_of_classes = len(classes)  # Liczba klas

    # Utworzenie słownika danych
    data = {
        'path': '/content/data',  # Ścieżka bazowa do danych
        'train': 'train/images',  # Ścieżka do obrazów treningowych
        'val': 'validation/images',  # Ścieżka do obrazów walidacyjnych
        'nc': number_of_classes,  # Liczba klas
        'names': classes  # Nazwy klas
    }

    # Zapisanie danych do pliku YAML
    with open(path_to_data_yaml, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f'Created config file at {path_to_data_yaml}')

    return

# Definicja ścieżki do pliku classes.txt i uruchomienie funkcji
path_to_classes_txt = '/content/custom_data/classes.txt'  # Ścieżka do pliku classes.txt
path_to_data_yaml = '/content/data.yaml'  # Ścieżka do pliku data.yaml

create_data_yaml(path_to_classes_txt, path_to_data_yaml)

# Wyświetlenie zawartości pliku data.yaml
print('\nFile contents:\n')
!cat /content/data.yaml
