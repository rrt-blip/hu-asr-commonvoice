# Hungarian ASR with Icefall/K2

## Project Description
This repository contains a Hungarian Automatic Speech Recognition (ASR) system implemented using the Icefall framework with k2. The system features:

- **Architecture**: Pruned Stateless Transducer with Zipformer encoder
- **Tokenization**: Byte Pair Encoding (BPE) with 500 subword units
- **Training Data**: Common Voice Hungarian corpus
- **Key Features**:
  - Achieves 14.85% WER on test set
  - Supports streaming inference
  - ONNX export compatible

Developed as part of BME project lab course.
