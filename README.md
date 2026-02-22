# HMM-Baum-Welch-Implementation
Implementation of the Baum–Welch algorithm for Hidden Markov Models with visualization of convergence, transition matrices, and emission probabilities.

# Hidden Markov Model (Baum-Welch) Visual Trainer

## Student Details
Name: Madhav Krishna T  
Register Number: TCR24CS043

---

## Project Overview

This project implements a Hidden Markov Model (HMM) trained using the Baum-Welch algorithm (Expectation-Maximization method).  
The system learns model parameters from an observed sequence and provides interactive visualizations of the training process.

The application runs as a local web dashboard built with:

• Python (Flask backend)  
• NumPy (numerical computation)  
• HTML, CSS, Bootstrap (UI)  
• Chart.js (graphs and visualization)  

---

## Features

### Training Module
• Accepts observed sequence as input  
• Accepts number of hidden states  
• Accepts number of observation symbols  
• Runs Baum-Welch iterations  
• Computes convergence automatically  

### Outputs Generated
• Transition matrix A  
• Emission matrix B  
• Initial distribution π  
• Probability of observation sequence P(O|λ)  
• Log-Likelihood at each iteration  
• Negative Log-Likelihood  
• Convergence statistics  

### Visualizations
• Log-Likelihood convergence graph  
• Observation probability graph  
• Optimization loss graph  
• Probability complement graph  
• State transition diagram  
• Heatmap matrices for A, B, and π  

---

## Project Structure

project/
│
├── app.py                → Flask backend with Baum-Welch implementation  
├── templates/index.html  → Dashboard UI  
├── static/ (optional)    → CSS/JS files  
└── README.md  

---

## How to Run the Project

1. Install dependencies

pip install flask numpy

2. Run the server

python app.py

3. Open browser and go to

http://127.0.0.1:5000

---

## How to Use

1. Enter an observed sequence  
   Example:  
   0 1 2 1 0 2 1 2  

2. Enter:
   • Number of hidden states (e.g. 5)  
   • Number of observation symbols (e.g. 3)  
   • Number of iterations (e.g. 100)  

3. Click **Train Model**

4. The dashboard will display:
   • Convergence graphs  
   • State transition diagram  
   • Learned matrices A, B, π  
   • Iteration log  

---

## Algorithm Description

This project uses the Baum-Welch algorithm, which is a special case of the Expectation-Maximization (EM) algorithm.

### Steps of the Algorithm

1. Initialize matrices A, B and vector π randomly  
2. Compute Forward probabilities α  
3. Compute Backward probabilities β  
4. Compute state probabilities γ  
5. Compute transition probabilities ξ  
6. Update A, B, and π using expectations  
7. Repeat until convergence  

---

## Mathematical Formulas

Forward:

α_t(j) = [ Σ_i α_{t−1}(i) * A_ij ] * B_j(O_t)

Backward:

β_t(i) = Σ_j A_ij * B_j(O_{t+1}) * β_{t+1}(j)

Parameter Updates:

A_ij = Σ_t ξ_t(i,j) / Σ_t γ_t(i)  
B_j(k) = Σ_{t : O_t = k} γ_t(j) / Σ_t γ_t(j)  
π_i = γ_1(i)

---

## Educational Purpose

This project demonstrates:

• Hidden state learning from data  
• Expectation-Maximization optimization  
• Convergence behaviour of probabilistic models  
• Visualization of parameter evolution  

---

## Future Improvements

• Viterbi decoding visualization  
• Multiple observation sequence support  
• Model saving/loading feature  
• Cloud deployment version  

---

## Submission Notes

• Repository must be public  
• Include README.md  
• Include source code  
• Provide instructions to run  

---

## Status

✔ Baum-Welch implementation complete  
✔ Visualization dashboard complete  
✔ Interactive training system complete  
✔ Ready for submission