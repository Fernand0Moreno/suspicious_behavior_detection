
Descripción del Proyecto
Este repositorio contiene el código y los recursos relacionados con el trabajo de investigación titulado "Detección de Comportamientos Sospechosos en Videovigilancia mediante Arquitecturas CNN-LSTM: Un Enfoque en Contextos Urbanos", realizado por Izurieta Pineda Ariana Shantal y Moreno Silva Kevin Fernando como parte de su titulación en Ingeniería en Ciencias de la Computación en la Universidad Politécnica Salesiana, Sede Guayaquil.

El proyecto propone un modelo híbrido basado en Redes Neuronales Convolucionales (CNN) y Redes de Memoria a Largo Corto Plazo (LSTM) para la detección automática de comportamientos sospechosos en videos de vigilancia urbana. El modelo fue evaluado utilizando el dataset público UCF-Crime, logrando resultados prometedores en la identificación de patrones temporales como agresiones, robos y forcejeos.

Características Principales
Arquitectura CNN-LSTM: Combina la extracción de características espaciales (CNN) con el modelado de dependencias temporales (LSTM).

Transfer Learning: Utiliza el modelo VGG16 pre-entrenado en ImageNet para mejorar la extracción de características.

Manejo de Desbalance de Clases: Implementa ponderación de clases durante el entrenamiento para abordar el desbalance en el dataset.

Evaluación Comparativa: Compara el rendimiento con modelos como YOLOv8, Random Forest y SVM.

Métricas de Evaluación: Incluye precisión, recall, F1-score, curvas ROC y matrices de confusión.

Estructura del Repositorio
text
suspicious_behavior_detection/
├── data_preprocessing/       # Scripts para preprocesamiento de datos
├── model/                    # Implementación del modelo CNN-LSTM
├── evaluation/               # Métricas y visualización de resultados
├── datasets/                 # Instrucciones para descargar datasets (UCF-Crime)
├── requirements.txt          # Dependencias de Python
└── README.md                 # Este archivo
Requisitos
Python 3.13.5

TensorFlow 2.x y Keras

Bibliotecas adicionales: OpenCV, NumPy, scikit-learn, Matplotlib, Pandas

Instalación
Clona el repositorio:

bash
git clone https://github.com/Fernand0Moreno/suspicious_behavior_detection.git
Instala las dependencias:

bash
pip install -r requirements.txt
Descarga el dataset UCF-Crime y colócalo en la carpeta datasets/.

Uso
Preprocesamiento de Datos:

bash
python data_preprocessing/preprocess.py
Entrenamiento del Modelo:

bash
python model/train.py
Evaluación:

bash
python evaluation/evaluate.py
Resultados
El modelo alcanzó una precisión del 92% en la detección de comportamientos sospechosos, superando a métodos tradicionales como YOLOv8 (85%). Para más detalles, consulta la sección de Resultados en el artículo científico.

Contribuciones
Implementación de un pipeline completo para detección de anomalías en videos.

Comparación detallada con otros modelos de aprendizaje automático.

Discusión sobre limitaciones y consideraciones éticas en videovigilancia automatizada.

Referencias
Para citar este trabajo, utiliza la siguiente referencia:

bibtex
@article{izurieta2025deteccion,
  title={Detección de Comportamientos Sospechosos en Videovigilancia mediante CNN-LSTM},
  author={Izurieta Pineda, Ariana Shantal and Moreno Silva, Kevin Fernando},
  year={2025},
  publisher={Universidad Politécnica Salesiana}
}
Licencia
Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.

Contacto
Ariana Shantal Izurieta Pineda: aizurieta@est.ups.edu.ec

Kevin Fernando Moreno Silva: kmorenos2@est.ups.edu.ec


