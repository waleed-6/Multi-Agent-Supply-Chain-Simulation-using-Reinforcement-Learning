# Multi-Agent Supply Chain Simulation using Reinforcement Learning

This project implements a decentralized, multi-agent supply chain environment designed for research and experimentation in Reinforcement Learning (RL). The system simulates real-world supply chain dynamics across multiple cities using intelligent agents that learn independently via Proximal Policy Optimization (PPO), powered by [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html).

## 📦 Project Overview

The environment is custom-built and modeled to reflect a multi-city supply chain (e.g., Jeddah and Riyadh) with four independent agents:

- **Inventory Agent**: Manages stock levels and replenishment.
- **Transport Agent**: Optimizes logistics and delivery scheduling.
- **Distribution Agent**: Controls allocation of goods across regions.
- **Adaptation Agent**: Dynamically adjusts operational parameters in response to system feedback.

All agents interact within a shared environment but maintain independent policies and rewards, enabling decentralized decision-making.

## 🎯 Key Features

- ✅ **Multi-agent system**: Designed with PettingZoo API to support multiple learning agents.
- ✅ **Custom supply chain environment**: Built from scratch with support for extensible state and action spaces.
- ✅ **Decentralized training**: Each agent is trained using independent PPO learners.
- ✅ **Modular design**: Easily extend or modify environment logic, reward structures, or agent roles.
- ✅ **Ray RLlib Integration**: Seamless integration with RLlib for scalable and configurable training.

## 🧱 Technologies Used

- Python 3.11
- [PettingZoo](https://www.pettingzoo.ml/)
- [Gymnasium](https://gymnasium.farama.org/)
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)
- NumPy, Pandas, Matplotlib (for data analysis)

## 🚀 How to Run

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/supply-chain-marl.git
   cd supply-chain-marl
   ```

2. **Install dependencies**  
   It's recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Train and test the agents**  
   ```bash
   python main.py
   ```


## 📂 Project Structure

```
supply-chain-marl/
├── env.py                      # Custom environment implementation
├── multi_wrapper.py            # wrapper in order to train the model
├── main.py                     # Training and testin the model
├── requirements.txt
└── README.md
```

## 📌 License

This project is released under the MIT License.

---

For research collaborations or inquiries, feel free to open an issue or contact the maintainer.
