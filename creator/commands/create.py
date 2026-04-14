import json
import os
import re
import sys
import uuid

import click
import questionary
from click import pass_context
from tinydb import Query, TinyDB

from creator.utils.cache import Cache
from creator.utils.style import STYLE


@click.command(
    "create",
    short_help="Create new simulation world",
)
@pass_context
def cli(ctx):
    cache = Cache()
    db = TinyDB(os.path.join(cache.worlds_path, "world_db.json"))

    # Lazy imports: keep CLI importable even if optional deps (e.g. openai) are missing.
    from creator.collections.utils import get_or_create_collection
    from creator.contexts_prompts.model import fmt_model_qa_tmpl
    from creator.contexts_prompts.world import fmt_world_qa_tmpl
    from creator.model_databases.fetch_worlds import download_world
    from creator.model_databases.objaverse import ObjaverseLoader
    from creator.sim_interfaces.mujoco import MujocoSimInterface
    from creator.utils.json import NumpyEncoder
    from creator.xml.worlds import find_model

    from creator.llm.model import prompt_model

    simulators = ["mujoco"]
    chosen_simulator = questionary.select(
        message=("Choose simulator to generate world for."),
        choices=simulators,
        style=STYLE,
    ).ask()

    chosen_model = "gpt-oss:120b-cloud"  # Gpt-4 is default and cheapest
    if chosen_simulator == "mujoco":
        loader = ObjaverseLoader()
        interface = MujocoSimInterface(chosen_model)
    models, worlds = loader.get_models()

    world_query = questionary.text(
        "Enter query for world generation(E.g Two cars and person next to it)",
        style=STYLE,
    ).ask()

    if not world_query:
        sys.exit(os.EX_OK)

    query = world_query.lower()
    query = query.replace("\n", "")

    World = Query()
    exists = db.search(World.prompt == query)

    if exists:
        questionary.print(
            f"World already exists at {exists[0]['filepath']}... 🦄",
            style="bold italic fg:green",
        )
        # return

    model_collection = get_or_create_collection("models_" + chosen_simulator, loader)
    try:
        claim_query_result = model_collection.query(
            query_texts=[query],
            include=["documents", "distances", "metadatas"],
            n_results=20,
        )

    except Exception:
        questionary.print(
            f"OpenAI api key at {cache.cache_path}/openai_api_key incorrect. "
            "Regenerate it at https://platform.openai.com/account/api-keys at copy "
            f"to {cache.cache_path}/openai_api_key",
            style="bold italic fg:red",
        )
        sys.exit(os.EX_DATAERR)

    context = [
        {"name": name, "metadata": metadata}
        for name, metadata in zip(
            claim_query_result["documents"][0], claim_query_result["metadatas"][0]
        )
    ]
    generate_world = False  # Pretty unstable, disabled for now

    if generate_world:
        content = fmt_world_qa_tmpl.format(context_str=worlds)

        questionary.print("Generating world... 🌎", style="bold fg:yellow")
        world = prompt_model(content, query, chosen_model)

        questionary.print(
            f"World is {world['World']}, downloading it", style="bold italic fg:green"
        )

        full_world = interface.find_world(world["World"], worlds)
        template_world_path = None
        if world["World"] != "None":
            template_world_path = download_world(
                world["World"], full_world["owner"], cache.worlds_path
            )

        if not template_world_path:
            questionary.print(
                "There was error in download world. Falling back to empty world",
                style="bold italic fg:red",
            )
            template_world_path = os.path.join(cache.worlds_path, "empty.sdf")
    else:
        template_world_path = os.path.join(cache.worlds_path, "empty.sdf")

    questionary.print(
        "Selecting models from database... 🫖", style="bold italic fg:yellow"
    )
    content = fmt_model_qa_tmpl.format(context_str=context)
    chosen_models = prompt_model(content, query, chosen_model)

    # Some models are hallucinated
    filtered_models = []
    for model in chosen_models:
        if find_model(model["Model"], models):
            filtered_models.append(model)
    chosen_models = filtered_models

    questionary.print(
        f"Placing {len(chosen_models)} models in the world... 📍",
        style="bold italic fg:yellow",
    )
    cleaned_query = re.sub(r'[<>:;.,"/\\|?*]', "", query).strip()
    world_name = f'world_{cleaned_query.replace(" ", "_")}'
    world_path = (
        os.path.join(cache.worlds_path, world_name) + interface.get_world_extension()
    )

    saved_models = interface.add_models(
        chosen_models, loader.get_models_full(), query, world_path, template_world_path
    )

    db.insert(
        {
            "id": str(uuid.uuid4()),
            "name": world_name,
            "filepath": world_path,
            "prompt": query,
            "total_models": json.dumps(saved_models, cls=NumpyEncoder),
            "world_name": "Empty",
        }
    )

    questionary.print(
        f"Finished! Output available at {world_path}",
        style="bold italic fg:green",
    )
