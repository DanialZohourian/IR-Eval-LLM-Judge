from openai import OpenAI

class Evaluator:
    def __init__(self, api_key: str, base_url: str, model: str = "gpt-4o-mini", prompt_template: str = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.prompt_template = prompt_template or """Here are a question and a retrieved passage from a text corpus from the same domain as the question.

Can you judge whether an answer to the question can be derived from the retrieved passage, simply answer either “YES” or “NO”.

<binary>

Question: {query}; Retrieved Passage: {document}"""
        self.relevance_dict = {}  # stores latest document->label
        self.relevance_labels = []  # stores latest labels only

    def get_relevance_labels(self, docs: list[dict], query: str, top_k=None) -> dict:
        """
        Use GPT-4o Mini to determine if each document is relevant to the query.
        Stores and returns a dictionary of {'document_text': 1 or 0}
        """
        self.relevance_dict = {}
        for i, doc in enumerate(docs[:top_k]):
            prompt = self.prompt_template.format(query=query, document=doc["document"])

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1
            )

            answer = response.choices[0].message.content.strip().upper()
            label = 1 if "YES" in answer else 0
            self.relevance_dict[doc["document"]] = label

            print(f"✅ [{i+1}/{len(docs[:top_k])}] Relevance: {answer}")

        self.relevance_labels = list(self.relevance_dict.values())
        return self.relevance_dict

    def compute_metrics(self, ks: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], show = False) -> dict:
        """
        Compute metrics using the most recent relevance labels.
        """
        if not self.relevance_labels:
            raise ValueError("No relevance labels available. Run get_relevance_labels() first.")

        metrics = {}
        total_relevant = sum(self.relevance_labels)

        for k in ks:
            topk = self.relevance_labels[:k]
            tp = sum(topk)

            precision = tp / k if k else 0
            recall = tp / total_relevant if total_relevant else 0
            accuracy = tp / k if k else 0
            binary_acc = 0 if tp == 0 else 1

            metrics[f"@{k}"] = {
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
                "Accuracy": round(accuracy, 3),
                "Number of Relevant Docs": tp,
                "Answer Presence": binary_acc
            }

        if show:
            for k, vals in metrics.items():
                print(f"\nMetrics {k}:")
                for metric, val in vals.items():
                    print(f"{metric}: {val}")

        return metrics
