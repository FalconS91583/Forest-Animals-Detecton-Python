# Importujemy niezbędne moduły
from pathlib import Path # Moduł do obsługi ścieżek plików i folderów
import random # Moduł do losowego wybierania elementów
import os # Moduł do operacji na systemie plików
import sys # Moduł do interakcji z systemem operacyjnym
import shutil # Moduł do kopiowania i przenoszenia plików
import argparse # Moduł do obsługi argumentów wiersza poleceń

# Definiujemy parser do obsługi argumentów wiersza poleceń
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', help='Path to data folder containing image and annotation files',
                    required=True) # Wymagany argument --datapath określający ścieżkę do folderu z danymi
parser.add_argument('--train_pct', help='Ratio of images to go to train folder; \
                    the rest go to validation folder (example: ".8")',
                    default=.8) # Opcjonalny argument określający procent danych treningowych (domyślnie 80%)

args = parser.parse_args() # Parsujemy argumenty podane przez użytkownika

# Pobieramy wartości argumentów
data_path = args.datapath # Ścieżka do folderu z danymi
train_percent = float(args.train_pct) # Procent danych treningowych jako liczba zmiennoprzecinkowa

# Sprawdzamy poprawność wprowadzonych danych
if not os.path.isdir(data_path): # Sprawdzamy, czy podana ścieżka istnieje i jest folderem
   print('Directory specified by --datapath not found. Verify the path is correct (and uses double back slashes if on Windows) and try again.')
   sys.exit(0) # Zakończenie programu, jeśli ścieżka jest niepoprawna
if train_percent < .01 or train_percent > 0.99: # Sprawdzamy, czy podany procent danych treningowych jest w dopuszczalnym zakresie
   print('Invalid entry for train_pct. Please enter a number between .01 and .99.')
   sys.exit(0) # Zakończenie programu, jeśli wartość jest nieprawidłowa
val_percent = 1 - train_percent # Obliczamy procent danych walidacyjnych jako resztę

# Definiujemy ścieżki do folderów wejściowych
input_image_path = os.path.join(data_path, 'images') # Ścieżka do folderu z obrazami
input_label_path = os.path.join(data_path, 'labels') # Ścieżka do folderu z plikami adnotacji

# Definiujemy ścieżki do folderów wyjściowych (treningowych i walidacyjnych)
cwd = os.getcwd() # Pobieramy aktualny katalog roboczy
train_img_path = os.path.join(cwd, 'data/train/images') # Ścieżka do folderu z obrazami treningowymi
train_txt_path = os.path.join(cwd, 'data/train/labels') # Ścieżka do folderu z plikami adnotacji treningowych
val_img_path = os.path.join(cwd, 'data/validation/images') # Ścieżka do folderu z obrazami walidacyjnymi
val_txt_path = os.path.join(cwd, 'data/validation/labels') # Ścieżka do folderu z plikami adnotacji walidacyjnych

# Tworzymy foldery wyjściowe, jeśli jeszcze nie istnieją
for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
   if not os.path.exists(dir_path): # Sprawdzamy, czy folder istnieje
      os.makedirs(dir_path) # Tworzymy folder, jeśli go nie ma
      print(f'Created folder at {dir_path}.')

# Pobieramy listę wszystkich plików obrazów i adnotacji
img_file_list = [path for path in Path(input_image_path).rglob('*')] # Lista wszystkich plików obrazów
txt_file_list = [path for path in Path(input_label_path).rglob('*')] # Lista wszystkich plików adnotacji

print(f'Number of image files: {len(img_file_list)}') # Wyświetlamy liczbę plików obrazów
print(f'Number of annotation files:  {len(txt_file_list)}') # Wyświetlamy liczbę plików adnotacji

# Obliczamy liczbę plików do przeniesienia do każdego folderu
file_num = len(img_file_list) # Całkowita liczba plików obrazów
train_num = int(file_num * train_percent) # Liczba plików obrazów do folderu treningowego
val_num = file_num - train_num # Liczba plików obrazów do folderu walidacyjnego
print(f'Images moving to train: {train_num}')
print(f'Images moving to validation: {val_num}')

# Losowo wybieramy pliki i przenosimy je do odpowiednich folderów
for i, set_num in enumerate([train_num, val_num]): # Iterujemy po liczbie plików dla treningu i walidacji
  for ii in range(set_num): # Iterujemy po liczbie plików w danej grupie
    img_path = random.choice(img_file_list) # Losowo wybieramy plik obrazu
    img_fn = img_path.name # Pobieramy nazwę pliku obrazu
    base_fn = img_path.stem # Pobieramy nazwę pliku bez rozszerzenia
    txt_fn = base_fn + '.txt' # Tworzymy nazwę odpowiadającego pliku adnotacji
    txt_path = os.path.join(input_label_path, txt_fn) # Tworzymy pełną ścieżkę do pliku adnotacji

    # Określamy ścieżki docelowe w zależności od grupy (trening lub walidacja)
    if i == 0: # Pliki treningowe
      new_img_path, new_txt_path = train_img_path, train_txt_path
    elif i == 1: # Pliki walidacyjne
      new_img_path, new_txt_path = val_img_path, val_txt_path

    # Kopiujemy plik obrazu do odpowiedniego folderu
    shutil.copy(img_path, os.path.join(new_img_path, img_fn))
    # Sprawdzamy, czy istnieje odpowiadający plik adnotacji, i kopiujemy go
    if os.path.exists(txt_path): # Jeśli plik adnotacji istnieje
      shutil.copy(txt_path, os.path.join(new_txt_path, txt_fn))

    # Usuwamy przeniesiony plik z listy dostępnych plików obrazów, aby uniknąć duplikatów
    img_file_list.remove(img_path)
