# CircBERT-GIN
Implementation of Circular miRNA Classification Model Based on DNABERT and GIN  
  CircBERT‑GIN is a deep learning framework that integrates DNABERT‑2 (sequence encoding), Graph Isomorphism Network (GIN) (RNA secondary structure graph encoding), and multi‑dimensional biological features for accurate classification of circulating miRNAs and potential biomarker discovery.  
The model fuses:  
  1. Sequence features from miRNA precursors  
  2. RNA secondary structure graph features  
  3. Genomic and thermodynamic biological attributes  
  4. Multi‑scale CNN + CBAM attention for feature enhancement  
  5. It achieves ~96.4% accuracy on the combined miRBase + GEO dataset for circulating miRNA identification.  
# Workflow
# Data collection  
  1. miRNA precursor sequences from miRBase  
  2. Circulating miRNA annotations from miRandola  
  3. miRNA‑Seq data from GEO (SRP253164, SRP275837, SRP288605, SRP324141, SRP372688)  
  4. Processed via miRDeep2 for novel miRNA detection & confidence scoring  
# CircBERT‑GIN model  
  1. DNABERT‑2: BPE + RoPE for long‑range sequence representation  
  2. GIN: One‑hot + EIIP + ANF encoding for RNA structure graphs  
  3. Multi‑branch CNN + CBAM: spatial & channel attention enhancement  
  4. MLP classifier for final binary prediction (circulating / non‑circulating)  
# Biomarker discovery  
  1. Feature importance by perturbation‑based masking  
  2. Top miRNA feature ranking  
  3. Functional enrichment analysis support  
# Files
  train.py: Train & test CircBERT‑GIN for classification   
  models.py: Full CircBERT‑GIN architecture  
  data_process.py: miRBase / GEO data parsing & preprocessing  
  feature.py: Sequence, structure, and biological feature extraction  
  train_test.py: Training loop, evaluation, cross‑validation  
  utils.py: Helper functions for encoding, attention, and visualization  
# Data
  Public datasets used in this study:  
  miRBase: https://mirbase.org/  
  miRandola: http://mirandola.di.unito.it/  
  GEO: https://www.ncbi.nlm.nih.gov/geo/  
  Human reference genome hg38  
# Data format
  Numeric feature matrix: (samples × features)  
  Sample IDs match label rows  
  Binary labels: 1 = circulating miRNA, 0 = non‑circulating  
# Requirements
  torch ≥ 1.10  
  cuda ≥ 11.3  
  python ≥ 3.7  
  transformers  
  torch_geometric  
  scikit‑learn  
  pandas, numpy, matplotlib  
 
