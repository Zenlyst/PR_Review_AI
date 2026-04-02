"""
Inference module — generate code reviews using the fine-tuned model.

Usage (from Colab notebook):
    from training.inference import generate_review, run_test_examples
    review = generate_review(model, tokenizer, before_code, after_code, file_path)
"""

import torch

from training.config import (
    GENERATION_MAX_TOKENS,
    GENERATION_REPETITION_PENALTY,
    GENERATION_TEMPERATURE,
    GENERATION_TOP_P,
    INFERENCE_TEMPLATE,
)


def generate_review(
    model,
    tokenizer,
    before_code: str,
    after_code: str,
    file_path: str = "unknown.py",
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """
    Generate a code review for a given before/after code pair.

    Args:
        model: Fine-tuned model (base + LoRA adapter).
        tokenizer: Tokenizer matching the model.
        before_code: Code before the change.
        after_code: Code after the change.
        file_path: File path for context (e.g., "utils/parser.py").
        max_new_tokens: Override default max generated tokens.
        temperature: Override default sampling temperature.

    Returns:
        Generated review comment as a string.
    """

    prompt = INFERENCE_TEMPLATE.format(
        file_path=file_path,
        before_code=before_code,
        after_code=after_code,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or GENERATION_MAX_TOKENS,
            temperature=temperature or GENERATION_TEMPERATURE,
            top_p=GENERATION_TOP_P,
            do_sample=True,
            repetition_penalty=GENERATION_REPETITION_PENALTY,
        )

    # Decode only the generated portion (skip the prompt)
    prompt_length = inputs["input_ids"].shape[1]
    review = tokenizer.decode(
        outputs[0][prompt_length:], skip_special_tokens=True
    )

    return review.strip()


def run_test_examples(model, tokenizer) -> None:
    """
    Run a set of test code review examples to evaluate the fine-tuned model.

    Prints the generated reviews for manual inspection.
    """

    test_cases = [
        {
            "name": "JSON → YAML migration without error handling",
            "file_path": "utils/parser.py",
            "before_code": (
                "def parse_config(path):\n"
                "    with open(path) as f:\n"
                "        data = json.load(f)\n"
                "    return data"
            ),
            "after_code": (
                "def parse_config(path):\n"
                "    with open(path) as f:\n"
                "        data = yaml.safe_load(f)\n"
                "    return data"
            ),
        },
        {
            "name": "SQL query without parameterization",
            "file_path": "db/queries.py",
            "before_code": (
                "def get_user(conn, user_id):\n"
                "    cursor = conn.cursor()\n"
                "    cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')\n"
                "    return cursor.fetchone()"
            ),
            "after_code": (
                "def get_user(conn, user_id):\n"
                "    cursor = conn.cursor()\n"
                "    cursor.execute(f'SELECT * FROM users WHERE id = {user_id} AND active = 1')\n"
                "    return cursor.fetchone()"
            ),
        },
        {
            "name": "Missing resource cleanup",
            "file_path": "services/downloader.py",
            "before_code": (
                "def download_file(url, dest):\n"
                "    response = requests.get(url)\n"
                "    with open(dest, 'wb') as f:\n"
                "        f.write(response.content)\n"
                "    return dest"
            ),
            "after_code": (
                "def download_file(url, dest):\n"
                "    response = requests.get(url, stream=True)\n"
                "    f = open(dest, 'wb')\n"
                "    for chunk in response.iter_content(8192):\n"
                "        f.write(chunk)\n"
                "    return dest"
            ),
        },
    ]

    print("=" * 60)
    print("🧪 Running Test Inference")
    print("=" * 60)

    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'─' * 60}")
        print(f"Test {i}: {tc['name']}")
        print(f"File: {tc['file_path']}")
        print(f"{'─' * 60}")

        review = generate_review(
            model=model,
            tokenizer=tokenizer,
            before_code=tc["before_code"],
            after_code=tc["after_code"],
            file_path=tc["file_path"],
        )

        print(f"\n📝 Generated Review:\n{review}")

    print(f"\n{'=' * 60}")
    print("✅ Test inference complete")
    print(f"{'=' * 60}")
