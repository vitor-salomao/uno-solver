# 🎉 UNO-Solver v1.0

![Release](https://img.shields.io/badge/release-v1.0-blue) ![Python](https://img.shields.io/badge/python-3.8%2B-yellow) [![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](./LICENSE)

**Created by Vitor Salomão & Andrew Sykes**

Welcome to **UNO-Solver**, your go-to CLI tool for finding the optimal UNO card play using deep reinforcement learning!



## 🚀 Features

- **🎯 Optimal Play**  
  Leverages a Deep Q-Network (DQN) to recommend the best move in any UNO hand scenario.

- **👥 4-Player Simulation**  
  Train and evaluate your AI in a realistic 4-player environment with custom opponent strategies.

- **⚡ Fast & Lightweight**  
  Built with PyTorch and Gym for quick inference—perfect for live hints or batch simulations.

- **🔧 Extensible**  
  Easily plug in new reward functions, opponent policies, or state encodings.



## 💾 Installation

```bash
# Clone the repo
git clone https://github.com/vitor-salomao/uno-solver.git
cd uno-solver

# Install dependencies
pip install -r requirements.txt
```



## 🎮 Usage

### Get a suggestion
```bash
python suggest.py \
    --hand "R5 Gskip B2" \
    --opp-cards "5 4 7" \
    --top-card "B7"
# Output: ▶️ Play B2 (Blue 2)
```

## 🤝 Contributing

We welcome changes! To contribute:
1. Fork this repo
2. Create a feature branch
3. Open a pull request

Please adhere to our code style and include tests for new functionality.


## 📝 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

*May the best card always be in your hand!* 🎴

