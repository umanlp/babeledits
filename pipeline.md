# BabelEdits: Pipeline description 

## Language Selection

- We select the 50 languages from XTREME-R. It's an easy choice since they are many and they have a good degree of typological diversity
- langs = ["af","ar","az","bg","bn","de","el","en","es","et","eu","fa","fi","fr","gu","he","hi","ht","hr","hu","id","it","ja","jv","ka","kk","ko","lt","ml","mr","ms","my","nl","pa","pl","pt","qu","ro","ru","sw","ta","te","th","tl","tr","uk","ur","vi","yo","zh"]

- Babelnet TIMEOUT params /work/tgreen/miniconda3/envs/babelnet/lib/python3.8/site-packages/babelnet/apis/rpc_api.py
## Entity Extraction

- There can be multiple strategies to extract entities from BabelNet:

    1. **Babelnet Traversal**:
        - you can use an iterator over BabelNet to get all the entities in the BabelNet graph. 
        - popularity measure: out-degree of the node or in-degree of the node. The latter requires a complete traversal of the graph.
        - Haven't explored this yet since it can be computationally expensive.
    2. **Wikipedia Querying per language** 
        - Wikipedia Page Selection (``get_pages.py``):
            - We selected a suitable time frame to extract the per-page view count. Currently using year 2021 (Bloom was released in mid-2022).
            - We can use the Wikipedia API to get the page view count for a certain language-specific wiki, we save the top-20k by view count.
            - We save for each page all its inter-language links together with the page view count.
            - We keep only pages that have an English title and remove duplicates.
            - We sort the results by number of inter-language links **filtered by selected languages** (to ensure higher degree of multi-parallelism) and keep the top-10k.
                - TODO compute Spearman correlation between view count and number of **filtered** inter-language links
                - One can create other versions of the datasets by sampling uniformly or taking tail entities
            - Time required: this takes 9 hours (532 mins), should be optimized (TODO)
        - Synset extraction (``get_synsets.py``):
            - for each language:
                - We extract the synsets from Babelnet using the English Wikipedia title and save them to disk.
                - We additionally save a frequency-sorted list of relations across all synsets.
                - This takes circa 16 hours
    3. **Wikipedia Querying per language + aggregation** [**CURRENTLY USED**]
        - Wikipedia Page Selection (``get_pages.py``):
            - Performed as above
        - Synset extraction:
            - GOAL: try to create multi-parallel edit dataset with K entities.
            - We create a single query list, which contains all the pages from all languages, uniformly sampled across the wikipedias (K / N_L entities per language).
                - This is done in ``merge_datasets.py`` which produces a single file (`all_langs.csv`)
            - We extract the synsets from Babelnet using the English Wikipedia title.
            - We keep a certain number of synsets, ensuring that each synsets covers all the languages we selected in Language Selection step.
            together with a frequency list of relations.
            - We update the per-language list of wikipedia pages to keep only those that "made it", i.e. those that will have a synset in Babelnet.

## Prompt Generation and Relation Selection

1. **Relation Aggregation** (`get_relations.py`): 
    - We aggregate the relations from all languages and sort them by frequency.
2. **Prompt Generation** (`get_relations.py`): 
    - For each relation in the top-200, we gather an example subject-predicate-object triple from the synsets.
    - We ask GPT-4 to generate a question for each relation.
    - EXAMPLE: EDUCATED_AT, Sushmita Sen, Mithibai College -->	Where was \<subject> educated?
3. **Relation Selection**:
    - We manually select the top-100 relations by frequency, together with subject and object and question

## Edit creation
- for each language l:
    - load the retrieved synsets
    - for each synset s: 
        - extract the edges which are in the selected relations set
        - we only keep an edge if the opposing synset has senses both in English and in the target language
        - for each edge e:
            - we randomly select a target synset from the list of synsets which have the same type of incoming edge as e
            - we extract the sense from that synset to make the edits
            - EXAMPLE: (Messi, PLAYS_FOR, Inter Miami) --> (Messi, PLAYS_FOR, Real Madrid)


## Translation

- We extract all the subject synsets
 from all the datasets and create a glossary (``get_glossary.py``) 
- This is uploaded to Google cloud storage (``upload_glossary.py``)
- We use the glossary to translate the edits to all the languages (``translate.py``)