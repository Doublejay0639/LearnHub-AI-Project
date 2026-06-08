import os
import json
from groq import Groq
from query import ask

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Test cases ────────────────────────────────────────────────────────────────
test_cases = [
    {
        "question": "What are the two types of user interface?",
        "ground_truth": "The two types of user interface are Command Line Interface (CLI) and Graphical User Interface (GUI).",
        "course": "H-C Interaction",
        "module": "Module 1 - Introduction",
    },
    {
        "question": "What is HCI?",
        "ground_truth": "HCI stands for Human-Computer Interaction. It is the study of how people interact with computers and to what extent computers are developed for successful interaction with human beings.",
        "course": "H-C Interaction",
        "module": "Module 1 - Introduction",
    },
    {
        "question": "What is usability?",
        "ground_truth": "Usability is one of the key concepts in HCI. It is concerned with making systems easy to learn and use.",
        "course": "H-C Interaction",
        "module": "Module 1 - Introduction",
    },
    {
        "question": "What are the golden rules of interface design?",
        "ground_truth": "The golden rules stated by Theo Mandel are: place the user in control, reduce the user's memory load, and make the interface consistent.",
        "course": "H-C Interaction",
        "module": "Module 1 - Introduction",
    },
    {
        "question": "What is GOMS?",
        "ground_truth": "GOMS stands for Goals, Operators, Methods, and Selection rules. It is a cognitive modeling framework used in HCI for analyzing and designing user interfaces.",
        "course": "H-C Interaction",
        "module": "Module 1 - Introduction",
    },
]

# ── Groq judge ────────────────────────────────────────────────────────────────
def judge(prompt: str) -> float:
    """Ask Groq to score something from 0.0 to 1.0. Returns the float."""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    raw = response.choices[0].message.content.strip()
    try:
        score = float(raw)
        return max(0.0, min(1.0, score))  # clamp to [0, 1]
    except ValueError:
        return 0.0


# ── Metric functions ──────────────────────────────────────────────────────────
def score_faithfulness(answer: str, sources: list) -> float:
    """Does the answer stick to what the retrieved sources say?"""
    source_names = ", ".join(sources) if sources else "none"
    prompt = f"""You are evaluating an AI tutor system.

The system retrieved these course files: {source_names}
The system produced this answer: "{answer}"

Score how faithful the answer is to typical university course material on this topic.
A score of 1.0 means the answer contains only factual, course-appropriate content.
A score of 0.0 means the answer contains hallucinations or made-up information.

Reply with ONLY a decimal number between 0.0 and 1.0. Nothing else."""
    return judge(prompt)


def score_answer_relevancy(question: str, answer: str) -> float:
    """Does the answer actually address the question asked?"""
    prompt = f"""You are evaluating an AI tutor system.

Question: "{question}"
Answer: "{answer}"

Score how well the answer addresses the question.
A score of 1.0 means the answer directly and completely answers the question.
A score of 0.0 means the answer is off-topic or does not address the question at all.

Reply with ONLY a decimal number between 0.0 and 1.0. Nothing else."""
    return judge(prompt)


def score_context_precision(question: str, sources: list, ground_truth: str) -> float:
    """Were the retrieved sources actually useful for answering this question?"""
    if not sources:
        return 0.0
    source_names = ", ".join(sources)
    prompt = f"""You are evaluating an AI tutor retrieval system.

Question: "{question}"
Expected answer topic: "{ground_truth}"
Retrieved files: {source_names}

Score how relevant the retrieved files are to answering this question.
A score of 1.0 means the retrieved files are highly relevant to the question.
A score of 0.0 means the retrieved files are completely irrelevant.

Reply with ONLY a decimal number between 0.0 and 1.0. Nothing else."""
    return judge(prompt)


# ── Run evaluation ────────────────────────────────────────────────────────────
print("Running LearnHub AI Evaluation...\n")

faithfulness_scores    = []
relevancy_scores       = []
context_precision_scores = []

for i, tc in enumerate(test_cases, 1):
    print(f"[{i}/{len(test_cases)}] {tc['question']}")

    result  = ask(question=tc["question"], course=tc["course"], module=tc["module"])
    answer  = result.get("answer", "")
    sources = [ref["file"] for ref in result.get("course_references", [])]

    f_score = score_faithfulness(answer, sources)
    r_score = score_answer_relevancy(tc["question"], answer)
    c_score = score_context_precision(tc["question"], sources, tc["ground_truth"])

    faithfulness_scores.append(f_score)
    relevancy_scores.append(r_score)
    context_precision_scores.append(c_score)

    print(f"  Faithfulness:      {f_score:.4f}")
    print(f"  Answer Relevancy:  {r_score:.4f}")
    print(f"  Context Precision: {c_score:.4f}")
    print()

# ── Final results ─────────────────────────────────────────────────────────────
avg_faithfulness    = sum(faithfulness_scores)    / len(faithfulness_scores)
avg_relevancy       = sum(relevancy_scores)       / len(relevancy_scores)
avg_context_precision = sum(context_precision_scores) / len(context_precision_scores)

print("=" * 50)
print("     LEARNHUB AI EVALUATION RESULTS")
print("=" * 50)
print(f"  Faithfulness:       {avg_faithfulness:.4f}  (target: > 0.80)")
print(f"  Answer Relevancy:   {avg_relevancy:.4f}  (target: > 0.80)")
print(f"  Context Precision:  {avg_context_precision:.4f}  (target: > 0.70)")
print("=" * 50)

# ── Save results to file ──────────────────────────────────────────────────────
results = {
    "faithfulness":      round(avg_faithfulness, 4),
    "answer_relevancy":  round(avg_relevancy, 4),
    "context_precision": round(avg_context_precision, 4),
    "per_question": [
        {
            "question":          test_cases[i]["question"],
            "faithfulness":      round(faithfulness_scores[i], 4),
            "answer_relevancy":  round(relevancy_scores[i], 4),
            "context_precision": round(context_precision_scores[i], 4),
        }
        for i in range(len(test_cases))
    ]
}

with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to eval_results.json")