from agent import run_agent

question = "Сколько суточных получит сотрудник за 4 дня командировки по стране?"
print(f"Вопрос: {question}\n")
answer = run_agent(question)
print(f"Ответ агента:\n{answer}")