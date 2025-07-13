
# Detección de Comportamientos Sospechosos en Videovigilancia mediante CNN-LSTM

![Python](https://img.shields.io/badge/Python-3.13.5-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Este repositorio contiene la implementación de un modelo híbrido CNN-LSTM para la detección automática de comportamientos sospechosos en videos de vigilancia urbana, desarrollado como parte de un trabajo de investigación en la Universidad Politécnica Salesiana, Sede Guayaquil.

## 📝 Descripción

El proyecto propone una arquitectura que combina Redes Neuronales Convolucionales (CNN) y Redes de Memoria a Largo Corto Plazo (LSTM) para identificar patrones temporales de comportamientos sospechosos (agresiones, robos, forcejeos) en videos de vigilancia. El modelo fue evaluado utilizando el dataset público UCF-Crime, demostrando resultados superiores a métodos tradicionales.

## ✨ Características Principales

- **Arquitectura CNN-LSTM**: Combina extracción de características espaciales con modelado de dependencias temporales
- **Transfer Learning**: Utiliza VGG16 pre-entrenado para mejorar la extracción de características
- **Manejo de datos desbalanceados**: Implementa ponderación de clases durante el entrenamiento
- **Evaluación comparativa**: Incluye comparación con YOLOv8, Random Forest y SVM
- **Métricas completas**: Precisión, recall, F1-score, curvas ROC y matrices de confusión

## 🗂 Estructura del Repositorio

suspicious_behavior_detection/
├── data_preprocessing/ # Scripts para preprocesamiento de datos
├── model/ # Implementación del modelo CNN-LSTM
├── evaluation/ # Métricas y visualización de resultados
├── datasets/ # Instrucciones para descargar datasets (UCF-Crime)
├── requirements.txt # Dependencias de Python
└── README.md # Documentación principal


## 🛠 Requisitos e Instalación

### 📋 Requisitos
- Python 3.13.5
- TensorFlow 2.x y Keras
- Bibliotecas adicionales: OpenCV, NumPy, scikit-learn, Matplotlib, Pandas

### ⚙️ Instalación
1. Clona el repositorio:
```bash
git clone https://github.com/Fernand0Moreno/suspicious_behavior_detection.git

### Instala las dependencias:

pip install -r requirements.txt

### Descarga el dataset UCF-Crime y colócalo en datasets/

### 🚀 Uso
Preprocesamiento de datos:
python data_preprocessing/preprocess.py

Entrenamiento del modelo:
python model/train.py

Evaluación:
python evaluation/evaluate.py

###  👥 Contribuciones
- Implementación de un pipeline completo para detección de anomalías en videos

- Comparación detallada con otros modelos de aprendizaje automático

- Análisis de limitaciones y consideraciones éticas en videovigilancia automatizada

### 📚 Referencias
Para citar este trabajo:

@article{izurieta2025deteccion,
  title={Detección de Comportamientos Sospechosos en Videovigilancia mediante CNN-LSTM},
  author={Izurieta Pineda, Ariana Shantal and Moreno Silva, Kevin Fernando},
  year={2025},
  publisher={Universidad Politécnica Salesiana}
}

### 📜 Licencia
Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.

### 📧 Contacto

- Ariana Shantal Izurieta Pineda: aizurieta@est.ups.edu.ec

- Kevin Fernando Moreno Silva: kmorenos2@est.ups.edu.ec

