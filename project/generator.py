import requests
import json
from trimesh.typed import Any
import prompts
from llm_request import ask_llm

model_name = "gpt-oss:120b-cloud"
base_url = "http://localhost:11434"
user_query = "гостиная с диваном, столом и креслами kuka_iiwa_14"

# расширяем запрос, для более насышенной сцены
def expand_prompt(prompt: str, model: str = model_name):
    expanded_prompt = ask_llm(
        prompt=prompt,
        system_prompt=prompts.expand_prompt_tmpl,
        model=model,
        timeout=60,
        base_url=base_url
    )
    return expanded_prompt

# Просто генерируем спискок объектов, которые должны быть в сцене, без привязки к конкретным моделям из ассет-базы
def generate_object_list(prompt: str, model: str = model_name):
    object_list = ask_llm(
        prompt=prompt,
        system_prompt=prompts.list_objects_tmpl,
        model=model,
        timeout=60,
        base_url=base_url
    )
    return object_list

# Приводим объекты в нормальный размер, чтобы соблюдались пропорции.
def normalize_object_scale():
    pass



# Формируем семантический план нашей сцены в виде графа со связами объектов между собой.)
def build_semantic_graph(
        prompt:str,
        object_list: list,
        model: str = model_name
    ):
    # Собираем промпт для генерации семантического плана, включающий в себя описание сцены и список объектов.
    content = 


    semantic_plan = ask_llm(
        prompt=prompt,
        system_prompt=prompts.semantic_graph_tmpl,
        model=model,
        timeout=60,
        base_url=base_url
    )
    return semantic_plan


def generate_asset(object_description: str, model: str = model_name):
    pass








def generate_scene(
    query: str,
    use_vlm_validation: bool = False,
    model: str = model_name,
):
    
    pass



def __main__():
    expanded_prompt = expand_prompt(prompt, model_name)
    print("Expanded Prompt:", expanded_prompt)


if __name__ == "__main__":
    __main__()