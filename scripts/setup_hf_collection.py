"""Create/refresh the BOTHROS Hugging Face Collection and add the model to it.

A Collection groups the weights under a searchable, script-named heading so a
researcher browsing or searching "Linear A" / "Linear B" finds it (the collection
title carries both script names, unlike the repo id "bothros"). Idempotent:
re-running won't duplicate the item.

    HF_TOKEN=$(cat ~/.hf_token) python3 release/scripts/setup_hf_collection.py
"""
import os

from huggingface_hub import (
    HfApi,
    add_collection_item,
    create_collection,
)

NAMESPACE = "JMacD263"
MODEL_REPO = "JMacD263/linear-a-linear-b-bothros"
TITLE = "BOTHROS — Linear A & Linear B sign reading"
DESC = (  # HF caps collection description at 150 chars
    "Read Aegean tablets from a photo: Linear A & Linear B signs by code + "
    "reading. YOLO + ConvNeXt, DAMOS/SigLA-benchmarked."
)


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (a WRITE token).")
    api = HfApi(token=token)

    # Reuse an existing collection with this title; else create it.
    slug = None
    for c in api.list_collections(owner=NAMESPACE, token=token):
        if c.title == TITLE:
            slug = c.slug
            print(f"Collection exists: {slug}")
            break
    if slug is None:
        col = create_collection(
            title=TITLE, namespace=NAMESPACE, description=DESC,
            private=False, token=token,
        )
        slug = col.slug
        print(f"Created collection: {slug}")

    add_collection_item(
        slug, item_id=MODEL_REPO, item_type="model",
        exists_ok=True, token=token,
    )
    print(f"Model in collection: {MODEL_REPO}")
    print(f"-> https://huggingface.co/collections/{slug}")


if __name__ == "__main__":
    main()
