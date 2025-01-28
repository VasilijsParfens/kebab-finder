from ultralytics import YOLO

def test_yolo():
    # Ielādē apmācīto modeli
    model = YOLO('models/trained_model.pt')  # Apmācītais modelis

    # Testa datu pārbaude
    results = model.predict(
        source='dataset/test',  # Ceļš uz testa attēliem
        save=True,              # Saglabā rezultātus `runs` mapē
        conf=0.5                # Uzticamības slieksnis
    )
    print("Testēšana pabeigta! Rezultāti saglabāti.")

if __name__ == "__main__":
    test_yolo()
