import pandas as pd

df = pd.read_json("agent_log.jsonl", lines=True)

print("=== СТАТИСТИКА И МЕТРИКИ АГЕНТА ===")

# Среднее количество шагов
steps = df[df.event == "step"].groupby("run_id").size()
print(f"Среднее число шагов на вопрос: {steps.mean():.2f}")

# Средняя задержка
if "latency_s" in df.columns:
    print(f"Средняя задержка LLM, сек: {df['latency_s'].mean():.2f}")

# Подсчет токенов
if "usage" in df.columns:
    tokens = df["usage"].apply(lambda x: x.get("total_tokens", 0) if isinstance(x, dict) else 0)
    print(f"Суммарное число токенов за все прогоны: {tokens.sum()}")
    print(f"Среднее число токенов на шаг: {tokens.mean():.2f}")

