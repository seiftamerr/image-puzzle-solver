# 🧩 Jigsaw Puzzle Solver using Classical Computer Vision

**Course:** CSE483 / CESS5004 – Computer Vision  
**Faculty:** Engineering, Ain Shams University

**Team Members**  
- Abdelrahman Khaled Gaber (23P0065)  
- Ahmed Walid Elsayed (23P0038)  
- Seif El Din Tamer (23P0240)  
- Omar Fouad (23P0146)

---

## 📌 Project Overview
This repository presents a **fully classical computer vision–based jigsaw puzzle solver** capable of reconstructing scrambled puzzles of different sizes **without using any machine learning or deep learning techniques**.

The project implements **three independent solvers**, each tailored to a specific puzzle size and complexity:

- **2×2 Solver** → Exhaustive edge matching with multi-channel descriptors  
- **4×4 Solver** → Greedy assembly with multi-anchor optimization  
- **8×8 Solver** → Progressive group-based assembly using global edge consistency

Each solver is designed with a **different algorithmic strategy**, demonstrating how problem size directly influences algorithm choice.

---

## ✨ Key Features
- 100% **classical computer vision** (OpenCV, NumPy, scikit-image)
- No training data required
- Scales from **4 to 64 pieces**
- Interpretable, step-by-step pipelines
- Quantitative evaluation using **MSE** and **SSIM**
- Visual demonstrations of reconstruction stages

---

## 🗂️ Repository Structure
```
Jigsaw_Puzzle_Solver/
├── Gravity Falls/
│   ├── correct/        # Ground truth images
│   ├── puzzle_2x2/     # Scrambled 2×2 puzzles
│   ├── puzzle_4x4/     # Scrambled 4×4 puzzles
│   └── puzzle_8x8/     # Scrambled 8×8 puzzles
│
├── milestone1/         # Early experiments & development stages
├── milestone2/         # Extended solvers and refinements
│
├── results/            # Output reconstructions & visualizations
│   ├── 2x2_out/
│   ├── 4x4_out/
│   └── 8x8_demo/
│
├── solver_2x2.py       # 2×2 exhaustive edge-based solver
├── solver_4x4.py       # 4×4 greedy multi-anchor solver
├── solver_8x8.py       # 8×8 group-based progressive solver
└── README.md
```
project/
├── solver_2x2.py      # Exhaustive edge-based solver
├── solver_4x4.py      # Greedy multi-anchor solver
├── solver_8x8.py      # Group-based progressive solver
├── data/
│   ├── puzzle_2x2/
│   ├── puzzle_4x4/
│   ├── puzzle_8x8/
│   └── correct/       # Ground truth images
├── results/
│   ├── 2x2_out/
│   ├── 4x4_out/
│   └── 8x8_demo/
└── README.md
```

---

## ⚙️ Installation & Setup

### Requirements
- Python 3.8+

### Install Dependencies
```bash
pip install opencv-python numpy matplotlib scikit-image scipy
```

---

## 🔍 Solver Pipelines

---

## 🔹 1. 2×2 Solver — Exhaustive Edge Matching

### Idea
With only **4 pieces**, the entire search space is small enough to allow **exhaustive evaluation of all 24 permutations**, guaranteeing a globally optimal solution.

### Pipeline
1. **Image Slicing**
   - Divide image into 4 equal quadrants

2. **Edge Feature Extraction**
   - Extract 3-pixel-wide edge strips
   - Compute a **5-channel descriptor**:
     - LAB color (3 channels)
     - Gradient magnitude (Sobel)
     - Laplacian response

3. **Edge Similarity Computation**
   - Standardize features (zero mean, unit variance)
   - Weighted distance:
     ```
     0.5 × LAB + 0.3 × Gradient + 0.2 × Laplacian
     ```

4. **Exhaustive Search**
   - Evaluate all 4! permutations
   - Select layout with minimum total edge cost
   - Confidence = (2nd best − best score)

5. **Validation**
   - Mean Squared Error (MSE)
   - Reconstruction correct if **MSE < 300**

### Properties
- Classical computer vision only
- Deterministic and fully interpretable
- Designed for educational and experimental use

---

## 🔹 2. 4×4 Solver — Greedy Multi-Anchor Assembly

### Idea
For **16 pieces**, exhaustive search is infeasible. Instead, a **greedy local optimization strategy** is used and strengthened via **multi-anchor restarts**.

### Pipeline
1. **Tile Extraction**
   - Slice image into 16 tiles (row-major order)

2. **Edge Cost Computation**
   - Use **SSD (Sum of Squared Differences)** between adjacent edges

3. **Greedy Assembly**
   - Fix one tile as the top-left anchor
   - Fill grid row-by-row
   - Each placement minimizes cost with left and top neighbors

4. **Multi-Anchor Optimization**
   - Repeat greedy assembly using all 16 tiles as anchors
   - Select arrangement with minimum total cost

5. **Validation**
   - Structural Similarity Index (SSIM)
   - Tile correct if **SSIM > 0.90**

### Properties
- Classical computer vision only
- Deterministic and fully interpretable
- Designed for educational and experimental use

---

## 🔹 3. 8×8 Solver — Progressive Group-Based Assembly

### Idea
For **64 pieces**, the solver mimics human puzzle solving by **progressively merging compatible pieces into larger groups** based on strong edge matches.

Unlike reference-based methods, this approach is **fully autonomous** and highly interpretable.

### Pipeline
1. **Piece Extraction**
   - Divide image into 64 pieces (28×28)
   - Random shuffle to simulate scrambling

2. **Color-Space Conversion**
   - Convert pieces to **LAB color space**

3. **Edge Cost Computation**
   - Compute horizontal (R→L) and vertical (B→T) edge costs using MSE

4. **Global Edge Ranking**
   - Collect all possible edge matches
   - Sort globally by increasing cost

5. **Progressive Group Merging**
   - Each piece starts as its own group
   - Iteratively merge groups if:
     - They do not overlap
     - Edge alignment is consistent

6. **Reconstruction**
   - Render the largest connected group
   - Shift placement to fit the 8×8 grid

7. **Validation**
   - Tile-level MSE
   - Tile correct if **MSE < 5**

### Properties
- Classical computer vision only
- Deterministic and fully interpretable
- Designed for educational and experimental use

---

---|--------|---------|--------------|
| 2×2 | 4 | Exhaustive | ~96% |
| 4×4 | 16 | Greedy + Multi-Anchor | ~91% |
| 8×8 | 64 | Group-Based Merging | High tile accuracy |

---

## 🚧 Limitations
- No rotation handling
- Grid-based pieces only
- Repetitive textures remain challenging
- No backtracking in greedy/group merges

---

## 🔮 Future Work
- Rotation-invariant matching
- Hierarchical backtracking for 8×8
- Irregular (non-grid) piece shapes
- Larger puzzles (10×10, 12×12)

---

## 📚 Technologies Used
- OpenCV
- NumPy
- scikit-image
- Matplotlib

---

## 📜 License
Educational project for academic use.

