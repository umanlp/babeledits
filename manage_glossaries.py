# %%
from google.cloud import translate

project_id = "babeledits-trial"
def list_glossaries(project_id: str = "YOUR_PROJECT_ID") -> translate.Glossary:
    """List Glossaries.

    Args:
        project_id: The GCP project ID.

    Returns:
        The glossary.
    """
    client = translate.TranslationServiceClient()

    location = "us-central1"

    parent = f"projects/{project_id}/locations/{location}"

    # Iterate over all results
    glossaries = []
    for glossary in client.list_glossaries(parent=parent):
        print(f"Name: {glossary.name}")
        print(f"Entry count: {glossary.entry_count}")
        print(f"Input uri: {glossary.input_config.gcs_source.input_uri}")

        # Note: You can create a glossary using one of two modes:
        # language_code_set or language_pair. When listing the information for
        # a glossary, you can only get information for the mode you used
        # when creating the glossary.
        for language_code in glossary.language_codes_set.language_codes:
            print(f"Language code: {language_code}")
        glossaries.append(glossary.name.split("/")[-1])

    return glossaries


def delete_glossary(
    project_id: str = "YOUR_PROJECT_ID",
    glossary_id: str = "YOUR_GLOSSARY_ID",
    timeout: int = 180,
) -> translate.Glossary:
    """Delete a specific glossary based on the glossary ID.

    Args:
        project_id: The ID of the GCP project that owns the glossary.
        glossary_id: The ID of the glossary to delete.
        timeout: The timeout for this request.

    Returns:
        The glossary that was deleted.
    """
    client = translate.TranslationServiceClient()

    name = client.glossary_path(project_id, "us-central1", glossary_id)

    operation = client.delete_glossary(name=name)
    result = operation.result(timeout)
    print(f"Deleted: {result.name}")

    return result

def delete_all(glossaries):
    for glossary in glossaries:
        print(f"Deleting glossary: {glossary}")
        delete_glossary(project_id, glossary)

print(list_glossaries(project_id))
# delete_all(list_glossaries(project_id))
# print(list_glossaries(project_id))
# %%
