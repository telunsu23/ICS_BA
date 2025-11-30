# Dynamic Stealthy Backdoor Attack Against Anomaly Detectors in ICS

This repository contains the official code implementation for the paper **"Dynamic Stealthy Backdoor Attack Against Anomaly Detectors in Industrial Control Systems"**.

## 📝 Introduction

Industrial Control Systems (ICS) are core components of modern critical infrastructure. While Deep Neural Network (DNN)-based anomaly detection methods have enhanced security, they also face threats from backdoor attacks. Existing backdoor methods typically use static triggers, which lack stealthiness in industrial scenarios and have limited attack effectiveness.

This project proposes a **dynamic stealthy backdoor attack** method tailored for ICS. This method achieves sample-adaptive trigger generation through an Encoder-Decoder architecture:

* **Encoder**: Encodes training set features into triggers and generates poisoned samples.

* **Decoder**: Reconstructs training set features from poisoned samples to ensure stealthiness.

* **Target**: Targets anomaly detection models.

## 📂 File Structure

```
ICS_BA/
├── ablation_study/       # Ablation study results (Lambda, Poison Ratio, Trigger Size)
├── baseline/             # Baseline comparison algorithms (TSBA, BackTime 等)
├── dataset/              # Dataset directory (BATADAL, HAI, SWaT)
│   ├── BATADAL/clean/    # Contains preprocessed scaler.pkl and hidden_info.json
│   └── ...
├── model/                # Model definitions
│   ├── decoder/          # Decoder network structure
│   ├── detector/         # Anomaly detector
│   └── encoder/          # Encoder network structure (Trigger Generator)
├── result/               # Experiment result save path
├── utils/                # Utility functions (Data loading, config reading, plotting)
├── config.yml            # Global configuration file
├── trigger_generator_train.py  # [Step 1] Train trigger generator
├── create_backdoored_dataset.py # [Step 2] Generate poisoned dataset
├── attack.py             # [Step 3] Execute backdoor attack (Train victim model)
├── attack_test.py        # [Step 4] Test attack effectiveness (ASR, BA)
└── ...
```

## 🛠️ Requirements

Please ensure the following dependencies are installed:

```
pip install -r requirements.txt
```

## 🚀 Quick Start

The attack process is mainly divided into three stages: training the trigger generator, generating poisoned data, and training/attacking the victim model.

### 1. Data Preparation

Please ensure the `dataset/` directory contains the cleaned data and standardization files (`scaler.pkl`) for the target datasets (BATADAL, SWaT, HAI).

### 2. Configuration Parameters

Modify the `config.yml` file to set experimental parameters:

* `dataset`: Select dataset ('BATADAL', 'SWaT', 'HAI')

* `poison_ratio`: Poisoning ratio (e.g., 0.1)

* `trigger_size`: Trigger size (alpha)

### 3. Run Attack Process

#### Step 1: Train Trigger Generator

Train the Encoder-Decoder network to learn how to generate stealthy and effective dynamic triggers.

```
python trigger_generator_train.py
```

* Model checkpoints will be saved in `result/<Dataset>/attack/trigger_generator.pth`。

#### Step 2: Create Backdoored Dataset

Use the trained generator to inject triggers into clean data.

```
python create_backdoored_dataset.py
```

#### Step 3: Launch Backdoor Attack Against Target Model

Train the anomaly detection model using the poisoned dataset.

```
python attack.py
```

#### Step 4: Evaluation

Evaluate Attack Success Rate (ASR) and Benign Accuracy (BA).

```
python attack_test.py
```


