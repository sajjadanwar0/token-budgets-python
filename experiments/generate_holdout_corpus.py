import json
import os
import sys
import time

N_PROMPTS = 100
META_PROMPT = (
    "Generate a diverse LLM agent prompt suitable for a tool-using "
    "assistant. Include the system prompt and user message together "
    "in your output (no preamble, no explanation). Make this prompt "
    "different from any standard example. Be specific about the "
    "task, the tools available, and the constraints. Aim for "
    "200-600 words of prompt text. Vary the domain across calls: "
    "code review, financial analysis, customer support, data "
    "transformation, scheduling, research synthesis, content "
    "moderation, multi-step research, technical writing, scientific "
    "calculation, etc."
)


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set")
    try:
        import openai
    except ImportError:
        sys.exit("ERROR: openai package not installed. pip install openai")
    client = openai.OpenAI()
    corpus = []
    t0 = time.time()
    for i in range(N_PROMPTS):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": META_PROMPT}],
                temperature=0.7,
                seed=42 + i,
            )
            text = resp.choices[0].message.content
        except Exception as e:
            print(f"  prompt {i}: ERROR {e}", file=sys.stderr)
            text = f"(ERROR: {e})"
        corpus.append({
            "idx": i,
            "seed": 42 + i,
            "prompt": text,
        })
        print(f"  {i+1}/{N_PROMPTS}: byte_len={len(text.encode('utf-8'))}")
    elapsed = time.time() - t0
    with open("holdout_corpus.json", "w") as f:
        json.dump(corpus, f, indent=2)
    total_bytes = sum(len(p["prompt"].encode("utf-8")) for p in corpus)
    print(f"\nWrote holdout_corpus.json")
    print(f"  Prompts: {len(corpus)}")
    print(f"  Total bytes: {total_bytes}")
    print(f"  Mean prompt length (bytes): {total_bytes / len(corpus):.0f}")
    print(f"  Wall time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()