import json
import re
from pathlib import Path


# ---------------------------------------------------------
# LOAD MASTER MAPPING
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MASTER_MAP_PATH = BASE_DIR / "master_map.json"


with open(MASTER_MAP_PATH, "r", encoding="utf-8") as f:
    MASTER_MAP = json.load(f)


# ---------------------------------------------------------
# KEYWORD MATCHING
# ---------------------------------------------------------

def _keyword_matches(question: str, keyword: str) -> bool:
    """
    Check whether a keyword/term occurs in the question.

    Matching is case-insensitive and uses word boundaries.
    """

    pattern = r"\b" + re.escape(keyword.lower()) + r"\b"

    return re.search(pattern, question.lower()) is not None


# ---------------------------------------------------------
# MAIN CHART DATA MAPPER
# ---------------------------------------------------------

def map_chart_keywords(question: str) -> dict:
    """
    Map keywords found in the question to chart entities.

    This function is intentionally deterministic.

    It does NOT:
    - understand intent
    - fetch chart data
    - query a database/vector store
    - generate embeddings
    - interpret astrology

    It simply checks the question against master_map.json
    and returns every matching mapped entity.
    """

    result = {
        "houses": [],
        "planets": [],
        "kp_specific": [],
        "divisional_charts": [],
        "timeframe": [],
    }

    if not question or not question.strip():
        return result

    question = question.strip()

    # ---------------------------------------------
    # Houses / Planets / KP / Divisional Charts
    # ---------------------------------------------

    for category in [
        "houses",
        "planets",
        "kp_specific",
        "divisional_charts",
    ]:

        mappings = MASTER_MAP.get(category, {})

        for entity, keywords in mappings.items():

            for keyword in keywords:

                if _keyword_matches(question, keyword):
                    result[category].append(entity)
                    break

    # ---------------------------------------------
    # Timeframe
    # ---------------------------------------------

    timeframe_keywords = MASTER_MAP.get("timeframe", [])

    for keyword in timeframe_keywords:

        if _keyword_matches(question, keyword):
            result["timeframe"].append(keyword)

    # ---------------------------------------------
    # Remove duplicates while preserving order
    # ---------------------------------------------

    for category in result:
        result[category] = list(dict.fromkeys(result[category]))

    return result


# ---------------------------------------------------------
# BUILD MAPPED RETRIEVAL QUERY
# ---------------------------------------------------------

def build_mapped_retrieval_query(
    question: str,
    mapper_output: dict
) -> str:
    """
    Build the temporary retrieval query.

    The original question is preserved.
    Mapper output is appended explicitly.

    This string is NOT stored in Pinecone.
    It is only used to generate the retrieval embedding.
    """

    parts = [question.strip()]

    if mapper_output.get("houses"):
        parts.append(
            "Relevant houses: "
            + ", ".join(mapper_output["houses"])
        )

    if mapper_output.get("planets"):
        parts.append(
            "Relevant planets: "
            + ", ".join(mapper_output["planets"])
        )

    if mapper_output.get("kp_specific"):
        parts.append(
            "Relevant KP concepts: "
            + ", ".join(mapper_output["kp_specific"])
        )

    if mapper_output.get("divisional_charts"):
        parts.append(
            "Relevant divisional charts: "
            + ", ".join(mapper_output["divisional_charts"])
        )

    if mapper_output.get("timeframe"):
        parts.append(
            "Relevant timeframe: "
            + ", ".join(mapper_output["timeframe"])
        )

    return "\n".join(parts)


# ---------------------------------------------------------
# LOCAL TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    test_questions = [
        "What about my career this year?",
        "What will Saturn do for my career in Pisces?",
        "Will I get married soon?",
        "What about my foreign travel and higher education?",
        "When will I get a promotion?",
        "What does my KP sub lord say?",
    ]

    for question in test_questions:

        mapper_output = map_chart_keywords(question)

        retrieval_query = build_mapped_retrieval_query(
            question,
            mapper_output
        )

        print("\n" + "=" * 70)
        print("QUESTION:")
        print(question)

        print("\nMAPPER OUTPUT:")
        print(json.dumps(mapper_output, indent=2))

        print("\nRETRIEVAL QUERY:")
        print(retrieval_query)