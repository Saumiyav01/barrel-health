# Barrel Health Monitoring
## Structural Health Monitoring and Remaining Useful Life Prediction Using Deep Learning

---

## 1. Project Overview

**Barrel Health Monitoring** is a prototype system designed to demonstrate the monitoring of barrel structural health through simulated condition monitoring and machine-learning-based Remaining Useful Life (RUL) estimation.

The project combines two independent technical components:

1. **Browser-Based Structural Health Monitoring Dashboard:** A web application developed using FastAPI that simulates cycle-by-cycle operating conditions, including temperature, vibration, stress, strain and overall health values. The generated readings are displayed through an interactive browser-based dashboard.

2. **Deep Learning-Based Predictive Maintenance Workflow:** A separate PyTorch-based machine-learning pipeline that prepares IMS bearing time-series data and trains a hybrid ResNet–BiLSTM architecture to estimate the Remaining Useful Life of a component.

The primary objective is to demonstrate how condition monitoring, time-series signal processing and deep learning can be used together as part of a predictive maintenance framework.

The dashboard demonstrates the visualization and monitoring aspect, while the machine-learning workflow explores data-driven degradation modelling and RUL estimation.

**Important:** These two components currently operate independently. The dashboard generates synthetic monitoring values and does not acquire live sensor measurements or receive predictions from the trained deep-learning model.

---

## 2. Problem Statement

Mechanical components operating under continuous or repetitive loading conditions can experience gradual degradation due to vibration, mechanical stress, temperature variations and accumulated operating cycles.

Traditional maintenance approaches generally depend on scheduled inspections or maintenance after a component has already developed a detectable fault.

These approaches may result in:

- Unexpected mechanical failures.
- Unnecessary maintenance operations.
- Increased equipment downtime.
- Reduced operational efficiency.
- Difficulty identifying gradual degradation.
- Limited information about the remaining useful life of components.

Predictive maintenance provides an alternative approach by analysing condition-monitoring data and identifying degradation patterns before complete component failure.

This project explores a prototype implementation combining simulated structural health monitoring and deep learning-based RUL estimation.

---

## 3. Project Objectives

The project is developed around the following objectives:

### 3.1 Structural Condition Monitoring

Simulate the continuous monitoring of important mechanical parameters, including:

- Temperature.
- Vibration.
- Mechanical stress.
- Mechanical strain.
- Overall health condition.

### 3.2 Cycle-Based Monitoring

Represent equipment operation through accumulated operating cycles and generate updated monitoring values as the simulated cycle count increases.

### 3.3 Interactive Visualization

Develop a browser-based dashboard that displays the latest monitoring values and health status, allowing users to observe changes during the simulation.

### 3.4 Time-Series Data Processing

Prepare raw IMS bearing time-series data for machine-learning applications through preprocessing, signal transformation and window-based dataset generation.

### 3.5 Deep Learning-Based RUL Estimation

Develop a hybrid ResNet–BiLSTM architecture that learns degradation-related patterns from time-series input data and estimates Remaining Useful Life.

### 3.6 Predictive Maintenance Exploration

Demonstrate how historical condition-monitoring data and deep learning can support maintenance planning by estimating the remaining operational life of a component.

---

## 4. System Architecture

The repository contains two separate workflows.

### 4.1 Overall Architecture

```text
                 BARREL HEALTH MONITORING
                           |
             +-------------+-------------+
             |                           |
             v                           v
     MONITORING DASHBOARD          MACHINE LEARNING
             |                           |
             v                           v
       FastAPI Backend              IMS Dataset
             |                           |
             v                           v
      Cycle Simulation             Data Preparation
             |                           |
             v                           v
     Simulated Parameters          Signal Processing
             |                           |
             v                           v
       Health Calculation          Window Generation
             |                           |
             v                           v
      API Response                 ResNet Feature Extraction
             |                           |
             v                           v
      Browser Dashboard            BiLSTM Sequence Learning
             |                           |
             v                           v
      Live Simulation View          RUL Estimation
                                         |
                                         v
                                  Model Checkpoint
```

**Note:** The diagram represents two independent implementation paths. There is currently no direct connection between the dashboard and the trained model.

### 4.2 Dashboard Architecture

The dashboard follows a client-server architecture.

**Frontend Layer**

The frontend is responsible for displaying monitoring information through a browser interface.

Technologies:
- HTML for the dashboard structure.
- JavaScript for dynamic updates and API communication.
- Charts for visualizing simulated monitoring parameters.

**Backend Layer**

The backend is developed using FastAPI.

Its responsibilities include:
- Initializing the monitoring simulation.
- Maintaining the simulated cycle count.
- Generating updated monitoring values.
- Returning the latest simulated readings.
- Providing API endpoints for frontend communication.

**Simulation Layer**

The simulation generates synthetic values for temperature, vibration, stress, strain and health based on the cycle-based monitoring workflow.

The generated values are returned to the browser through API responses.

### 4.3 Machine-Learning Architecture

The machine-learning workflow follows a separate data-processing and training pipeline.

```text
       Raw IMS Time-Series Data
                  |
                  v
          Dataset Preparation
                  |
                  v
           Signal Processing
                  |
                  v
         Feature Transformations
                  |
                  v
           Window Generation
                  |
                  v
             ResNet Model
          Feature Extraction
                  |
                  v
             BiLSTM Model
          Sequence Learning
                  |
                  v
            RUL Estimation
                  |
                  v
          Model Checkpoint
```

The hybrid architecture combines convolutional feature extraction with sequential modelling to learn degradation-related patterns from time-series data.

---

## 5. Technologies and Tools Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Main programming language |
| FastAPI | Backend API development |
| Uvicorn | ASGI application server |
| HTML | Dashboard structure |
| JavaScript | Dynamic dashboard updates |
| Jinja2 | HTML template rendering |
| NumPy | Numerical computation and dataset arrays |
| SciPy | Scientific and signal-processing operations |
| Scikit-learn | Machine-learning utilities and preprocessing |
| PyTorch | Deep learning model development and training |
| ResNet | Time-series feature extraction |
| BiLSTM | Sequential pattern learning |
| IMS Dataset | Bearing degradation time-series data |

---

## 6. Browser-Based Monitoring Dashboard

### 6.1 Introduction

The monitoring dashboard provides a graphical interface for observing simulated barrel operating conditions.

It is served through a FastAPI backend and accessed using a web browser.

Users can enter the number of operating cycles already completed and initiate the monitoring simulation.

After initialization, the dashboard periodically requests updated information from the backend and displays the generated monitoring values.

### 6.2 Monitoring Parameters

The dashboard simulates five important monitoring parameters.

**Temperature**

Represents the simulated thermal condition of the component.

Temperature monitoring is important because changes in operating temperature may be associated with mechanical friction, loading conditions or thermal degradation.

**Vibration**

Represents the simulated vibration behaviour of the component.

Vibration is an important condition-monitoring parameter because abnormal vibration patterns can indicate mechanical imbalance, looseness, bearing-related problems or other mechanical abnormalities.

**Stress**

Represents the simulated mechanical stress experienced by the component.

Stress describes the internal force intensity developed within a material due to applied loading.

**Strain**

Represents the simulated deformation response of the component under mechanical loading.

Strain describes the relative deformation of a material compared with its original dimensions.

**Health Value**

Represents a simulated overall health indicator used by the dashboard to communicate the component's condition.

It provides a simplified representation of the component's simulated health status.

These values are generated by the simulation rather than measured directly from physical sensors.

### 6.3 Dashboard Operation

The monitoring process follows these steps:

1. The user opens the dashboard through the browser.
2. The user enters the number of operating cycles already completed.
3. The frontend sends the cycle count to the FastAPI backend.
4. The backend initializes or resets the simulation.
5. The frontend periodically requests the latest monitoring information.
6. The backend advances the simulation by one cycle.
7. Updated temperature, vibration, stress, strain and health values are returned.
8. The dashboard refreshes its displayed readings and charts.

This process continues while the monitoring interface is active.

### 6.4 Dashboard Technologies

**FastAPI:** Handles incoming HTTP requests and generates monitoring responses.

**Uvicorn:** Runs the FastAPI application as an ASGI server.

**Jinja2:** Provides HTML template rendering.

**JavaScript:** Handles API polling and dynamic updates.

**HTML:** Defines the browser-based monitoring interface.

---

## 7. FastAPI Backend and API Endpoints

The backend provides three primary endpoints.

| HTTP Method | Endpoint | Function |
|---|---|---|
| GET | `/` | Serves the monitoring dashboard |
| POST | `/initialize` | Initializes or resets the simulation |
| GET | `/latest` | Advances the simulation by one cycle and returns the latest readings |

### 7.1 Dashboard Endpoint

**Endpoint:** `GET /`

This endpoint serves the main monitoring dashboard.

When the user accesses the root URL, FastAPI renders the dashboard page using the configured HTML template.

### 7.2 Initialize Endpoint

**Endpoint:** `POST /initialize`

This endpoint initializes the simulation using the operating cycle count supplied by the user.

Example request:

```json
{
    "cycles": 2000
}
```

The backend uses the supplied value as the starting cycle count for the simulation.

This allows the user to simulate a component that has already undergone a specified number of operating cycles.

### 7.3 Latest Readings Endpoint

**Endpoint:** `GET /latest`

This endpoint advances the simulation by one cycle and returns the latest simulated monitoring values.

The returned information represents the updated simulation state.

The frontend uses this endpoint to retrieve information and refresh the dashboard.

**Important:** Calling this endpoint changes the simulation state because each request advances the simulated cycle count.

---

## 8. Machine-Learning Workflow

### 8.1 Introduction

The machine-learning component focuses on estimating Remaining Useful Life using time-series degradation data.

It uses the IMS bearing dataset as its input source.

The workflow is implemented using PyTorch and consists of dataset preparation, signal preprocessing, window generation, model training and experimental inference.

The raw IMS dataset is not included in the repository and must be supplied separately.

### 8.2 IMS Dataset

The IMS bearing dataset contains time-series measurements associated with bearing operation and degradation.

It is used in this project to develop and train a model capable of learning patterns from mechanical condition-monitoring signals.

The dataset is expected to be placed inside:

```text
data/
└── raw/
    └── ims/
        ├── IMS data files
        └── ...
```

The dataset preparation script reads the supplied raw files and generates arrays for model training.

### 8.3 Dataset Preparation

The dataset preparation process is performed using:

```bash
python build_dataset.py
```

The script processes the raw IMS data and generates three primary NumPy arrays.

| Output File | Purpose |
|---|---|
| `dataset_windows.npy` | Prepared time-series windows used as model inputs |
| `dataset_stats.npy` | Statistical information generated during dataset preparation |
| `dataset_rul.npy` | Corresponding RUL target values |

These files are saved in the repository root.

**Window Generation**

Time-series data is divided into smaller segments called windows.

Instead of processing an entire signal as a single input, the model receives smaller portions of the original sequence.

This allows the training pipeline to learn patterns from different portions of the operating history.

**Preprocessing**

The preprocessing stage prepares the signals for model training.

The repository includes preprocessing modules and physics-informed feature transformations.

These transformations are intended to extract useful information from the original time-series signals.

The exact transformations and parameters are defined in the preprocessing implementation.

---

## 9. Hybrid ResNet–BiLSTM Architecture

### 9.1 Model Overview

The primary deep-learning architecture is a hybrid ResNet–BiLSTM model implemented using PyTorch.

The model combines two neural-network approaches:

- Residual convolutional learning using ResNet.
- Sequential learning using Bidirectional Long Short-Term Memory (BiLSTM).

The purpose of combining these approaches is to learn both signal-related features and temporal degradation patterns.

### 9.2 ResNet – Feature Extraction

ResNet is a residual neural-network architecture.

It uses shortcut connections that allow information to pass through network layers without being transformed by every intermediate layer.

These connections help train deeper neural networks and improve feature learning.

In this project, the ResNet component is used to extract useful representations from the prepared time-series input.

Potentially useful signal characteristics include:
- Changes in signal amplitude.
- Local temporal patterns.
- Variations in vibration behaviour.
- Degradation-related signal characteristics.

### 9.3 BiLSTM – Sequential Learning

BiLSTM stands for Bidirectional Long Short-Term Memory.

It is a recurrent neural-network architecture designed to learn relationships in sequential data.

A conventional LSTM processes a sequence in one direction, whereas a BiLSTM processes the sequence in both forward and backward directions.

This allows the network to learn contextual relationships from both directions within the input sequence.

In this project, BiLSTM is used as part of the hybrid architecture to learn temporal relationships in the extracted feature representations.

### 9.4 Remaining Useful Life Estimation

Remaining Useful Life (RUL) refers to the estimated operating life remaining before a component reaches its defined end-of-life condition.

The model is trained to produce a scalar RUL estimate.

The predicted value can be used as an experimental indicator of the remaining operating life represented by the training data.

The model architecture is defined in:

```text
models/
└── resnet_bilstm.py
```

---

## 10. Model Training

After preparing the dataset, the model is trained using:

```bash
python train.py
```

The training workflow uses the generated NumPy arrays as input data.

The process includes:
1. Loading the prepared dataset.
2. Preparing model inputs and corresponding RUL targets.
3. Initializing the hybrid ResNet–BiLSTM model.
4. Performing forward propagation.
5. Calculating the training loss.
6. Updating model parameters through backpropagation.
7. Evaluating model performance according to the implemented training procedure.
8. Saving the best model checkpoint.

The best checkpoint is saved as:

```text
best_rul_model.pth
```

This checkpoint contains the trained model information required by the corresponding experimental RUL workflow.

### GPU Training

PyTorch supports GPU acceleration when an appropriate CUDA-compatible environment is configured.

For GPU-based training, install the PyTorch version compatible with the available hardware and CUDA configuration.

Official installation guide:

https://pytorch.org/get-started/locally/

---

## 11. Additional Scripts

### 11.1 predict.py

This script provides an experimental inference workflow for an IMS-format signal file.

It attempts to load:

```text
best_hybrid_model.pth
```

The script currently interprets the model output as a wear-stage classification result.

However, the shared model architecture produces a scalar RUL estimate rather than a wear-stage class.

This creates an incompatibility between the inference interpretation and the model output.

Consequently, the predicted wear-stage result should not be considered reliable until the model output and inference implementation are made compatible.

The example input path in the script must also be updated to point to the intended IMS-format signal file.

### 11.2 deployment_simulation.py

This script demonstrates a synthetic cycle-based RUL simulation.

It uses:

```text
best_rul_model.pth
```

The script explores an experimental deployment-style demonstration in which operating cycles and RUL-related values are represented synthetically.

It should be distinguished from the FastAPI dashboard because the two scripts perform separate functions.

---

## 12. Installation and Setup

### 12.1 System Requirements

| Requirement | Minimum |
|---|---|
| Operating System | Windows / Linux / macOS |
| Python | 3.10 or later |
| Package Manager | pip |
| Internet | Required for package installation |
| GPU | Optional |

### 12.2 Clone the Repository

```bash
git clone <repository-url>
```

Navigate into the project directory:

```bash
cd <repository-folder>
```

### 12.3 Install Dependencies

Install the required Python packages:

```bash
python -m pip install fastapi "uvicorn[standard]" jinja2 numpy scipy scikit-learn torch
```

For GPU training, follow the official PyTorch installation instructions.

### 12.4 Start the Monitoring Dashboard

From the repository root, execute:

```bash
uvicorn app:app --reload
```

After the server starts, open the following address:

http://127.0.0.1:8000

Enter the number of cycles already used and select **Start Monitoring**.

The dashboard will begin displaying simulated
