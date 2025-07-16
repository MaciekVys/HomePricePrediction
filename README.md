# 🏡 Home Price Prediction – Machine Learning Project

Projekt regresyjny mający na celu przewidywanie cen domów na podstawie wybranych cech. Został zbudowany przy użyciu modelu **Gradient Boosting Regressor**, wzbogaconego o **strojenie hiperparametrów** oraz **API do predykcji z Flask**.

---

## 🔍 Cel projektu

Celem projektu było:

- Zbudowanie modelu regresyjnego przewidującego ceny domów.
- Porównanie różnych algorytmów ML (KNN, Decision Tree, Random Forest, Gradient Boosting).
- Zastosowanie strojenia hiperparametrów (GridSearchCV).
- Utworzenie REST API z Flask do obsługi predykcji.
- Przygotowanie profesjonalnej dokumentacji.

---

## 🧠 Wykorzystane algorytmy

| Model                    | MSE ↓          | R² Score ↑ |
|--------------------------|----------------|------------|
| KNN                      | 20.66          | 0.72       |
| Decision Tree            | 11.14          | 0.85       |
| Logistic Regression      | 24.29          | 0.67       |
| Random Forest Regressor  | 8.10           | 0.89       |
| Gradient Boosting (tuned)| **6.24**       | **0.91**   |

---

## ⚙️ Strojenie hiperparametrów

Użyto `GridSearchCV` z parametrami:

```python
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'max_depth': [1, 2, 3, 4, 5, 7]
}
```
---
## 📚 Technologie

- Python 3.12
- scikit-learn
- pandas, numpy, matplotlib, seaborn
- Flask
- GridSearchCV
- Jupyter Notebook

## ✅ Wnioski
- Gradient Boosting dał najlepsze wyniki.
- Hiperparametry znacząco poprawiły skuteczność modelu.
- Cechy które znacząco wpływają na cenę domu to : Średnia liczba pokoi na mieszkanie, procent mieszkańców o niskim statusie społecznym, uśredniona odległość do 5 głównych centrów zatrudnienia w Bostonie i Przestępczość na jednego mieszkańca.

## 📦 Instalacja

```bash
git clone https://github.com/MaciekVys/HomePricePrediction.git
cd HomePricePrediction
pip install -r requirements.txt
python app.py
```
