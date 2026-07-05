# check_dataset.py
from pathlib import Path

data_path = Path("T:/recognize_anomaly/data/ucf_crime")

print("=== Проверка структуры датасета ===\n")

if not data_path.exists():
    print(f"❌ Папка {data_path} не существует!")
    exit(1)

# Проверяем папки
anomaly_path = data_path / "Anomaly_Videos"
normal_path = data_path / "Normal_Videos"
test_path = data_path / "test"

if anomaly_path.exists():
    subdirs = [d for d in anomaly_path.iterdir() if d.is_dir()]
    print(f"✅ Anomaly_Videos: {len(subdirs)} подпапок")
    total = sum(len(list(d.glob("*.mp4"))) for d in subdirs)
    print(f"   - Всего видео: {total}")
else:
    print("❌ Anomaly_Videos не найдена!")

if normal_path.exists():
    count = len(list(normal_path.glob("*.mp4")))
    print(f"✅ Normal_Videos: {count} видео")
else:
    print("❌ Normal_Videos не найдена!")

if test_path.exists():
    files = list(test_path.glob("*.txt"))
    print(f"✅ test: {len(files)} файлов разметки")
else:
    print("❌ test не найдена!")

print("\n=== Если всё зелёное, можно продолжать ===")