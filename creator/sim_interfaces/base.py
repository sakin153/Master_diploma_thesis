from typing import Dict, List

from creator.contexts_prompts.place import fmt_place_qa_tmpl
from creator.contexts_prompts.scale import fmt_scale_qa_tmpl
from creator.llm.model import prompt_model


class BaseSimInterface:
    def __init__(self, chosen_model):
        self.chosen_model = chosen_model

    def prompt_model_for_scale(
        self, models_for_scale: List[Dict], query: str
    ) -> List[Dict]:
        content = fmt_scale_qa_tmpl.format(models_for_scaling=str(models_for_scale))
        return prompt_model(content, query, self.chosen_model)

    def prompt_model_for_placement(
        self, models_for_placement: List[Dict], query: str
    ) -> List[Dict]:
        content = fmt_place_qa_tmpl.format(
            context_str=f"Arrange following models: {str(models_for_placement)}",
        )
        return prompt_model(content, query, self.chosen_model)

    def prompt_model_for_constraints(self, content: str, query: str) -> Dict:
        return prompt_model(content, query, self.chosen_model)
