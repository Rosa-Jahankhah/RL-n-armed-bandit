import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    """
    An n-armed bandit where each arm has a true value drawn from N(0,1).
    Reward for pulling arm a ~ N(q(a), 1).
    """

    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        # True action values q(a) — hidden from the agent
        self.q_true = np.random.randn(n_arms)
        self.optimal_arm = np.argmax(self.q_true)

    def pull(self, arm):
        """Return a noisy reward from the chosen arm."""
        return np.random.randn() + self.q_true[arm]


class EpsilonGreedyAgent:
    """
    Epsilon-greedy agent that estimates action values by sample averaging.
    epsilon=0.0  -> pure greedy (exploit only)
    epsilon=0.1  -> 10% random exploration
    epsilon=1.0  -> pure random
    """

    def __init__(self, n_arms, epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.Q = np.zeros(n_arms)   # estimated action values
        self.N = np.zeros(n_arms)   # how many times each arm was pulled

    def select_action(self):
        """Choose an arm using epsilon-greedy rule."""
        if np.random.rand() < self.epsilon:
            # Explore: pick a random arm
            return np.random.randint(self.n_arms)
        else:
            # Exploit: pick the arm with highest estimated value
            # np.argmax breaks ties by returning the first maximum
            return np.argmax(self.Q)

    def update(self, arm, reward):
        """Update the sample average estimate for the chosen arm."""
        self.N[arm] += 1
        # Incremental update rule: Q(a) <- Q(a) + (1/N) * (R - Q(a))
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]


def run_experiment(n_arms=10, n_steps=1000, n_runs=2000, epsilons=(0.0, 0.01, 0.1)):
    """
    Run multiple independent bandit tasks and average the results.
    Returns average reward and % optimal action per step for each epsilon.
    """
    results = {}

    for eps in epsilons:
        avg_rewards = np.zeros(n_steps)
        pct_optimal = np.zeros(n_steps)

        for _ in range(n_runs):
            bandit = Bandit(n_arms)
            agent = EpsilonGreedyAgent(n_arms, eps)

            for t in range(n_steps):
                arm = agent.select_action()
                reward = bandit.pull(arm)
                agent.update(arm, reward)

                avg_rewards[t] += reward
                if arm == bandit.optimal_arm:
                    pct_optimal[t] += 1

        results[eps] = {
            "avg_reward": avg_rewards / n_runs,
            "pct_optimal": (pct_optimal / n_runs) * 100,
        }

    return results


def plot_results(results):
    """Reproduce Figure 2.1 from Sutton & Barto."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    colors = {0.0: "#E24B4A", 0.01: "#1D9E75", 0.1: "#378ADD"}

    for eps, data in results.items():
        label = f"ε = {eps}"
        ax1.plot(data["avg_reward"], label=label, color=colors[eps], linewidth=1.5)
        ax2.plot(data["pct_optimal"], label=label, color=colors[eps], linewidth=1.5)

    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average reward")
    ax1.set_title("Average reward over time (10-armed bandit, 2000 runs)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal action")
    ax2.set_title("% Optimal action over time")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("Plot saved to results.png")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    print("Running 2000 bandit experiments × 1000 steps each...")
    results = run_experiment()
    plot_results(results)
    print("Done!")