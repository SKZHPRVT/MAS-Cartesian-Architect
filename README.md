# MAS Cartesian Architect

Система мультиагентной координации с квадратом Декарта.

## Установка
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Запуск

Демонстрация концепции (7 групп):
```bash
python 01_demo_concept.py
```

Поиск ординара (70 групп, 10 поколений):
```bash
python 02_search_ordinal.py
```

Диагностика (50 групп):
```bash
python 03_debug_diagnostic.py
```

Стресс-тест:
```bash
python 04_stress_test.py
```

## Стратегии
greedy(-+), genetic(+-), swarm(++), hybrid(++), auction(++/+-), ml_based(++), random(--)
