# CircBERT-GIN
Implementation of Circular miRNA Classification Model Based on DNABERT and GIN
Overview
  CircBERT‑GIN is a deep learning framework that integrates DNABERT‑2 (sequence encoding), Graph Isomorphism Network (GIN) (RNA secondary structure graph encoding), and multi‑dimensional biological features for accurate classification of circulating miRNAs and potential biomarker discovery.
The model fuses:
  Sequence features from miRNA precursors
  RNA secondary structure graph features
  Genomic and thermodynamic biological attributes
  Multi‑scale CNN + CBAM attention for feature enhancement
  It achieves ~96.4% accuracy on the combined miRBase + GEO dataset for circulating miRNA identification.
Workflow
Data collection
  miRNA precursor sequences from miRBase
  Circulating miRNA annotations from miRandola
  miRNA‑Seq data from GEO (SRP253164, SRP275837, SRP288605, SRP324141, SRP372688)
  Processed via miRDeep2 for novel miRNA detection & confidence scoring
CircBERT‑GIN model
  DNABERT‑2: BPE + RoPE for long‑range sequence representation
  GIN: One‑hot + EIIP + ANF encoding for RNA structure graphs
  Multi‑branch CNN + CBAM: spatial & channel attention enhancement
  MLP classifier for final binary prediction (circulating / non‑circulating)
Biomarker discovery
  Feature importance by perturbation‑based masking
  Top miRNA feature ranking
  Functional enrichment analysis support
Files
  train.py: Train & test CircBERT‑GIN for classification
  main_biomarker: Identify circulating miRNA biomarkers
  models.py: Full CircBERT‑GIN architecture
  data_process.py: miRBase / GEO data parsing & preprocessing
  feature_engineering.py: Sequence, structure, and biological feature extraction
  train_test.py: Training loop, evaluation, cross‑validation
  utils.py: Helper functions for encoding, attention, and visualization
Data
  Public datasets used in this study:
  miRBase: https://mirbase.org/
  miRandola: http://mirandola.di.unito.it/
  GEO: https://www.ncbi.nlm.nih.gov/geo/
  Human reference genome hg38
Data format
  Numeric feature matrix: (samples × features)
  Sample IDs match label rows
  Binary labels: 1 = circulating miRNA, 0 = non‑circulating
Requirements
  torch ≥ 1.10
  cuda ≥ 11.3
  python ≥ 3.7
  transformers
  torch_geometric
  scikit‑learn
  pandas, numpy, matplotlib
Usage
  Train and evaluate the model
  bash python main_circbertgin.py
  Discover circulating miRNA biomarkers
  bash python main_biomarker.py
