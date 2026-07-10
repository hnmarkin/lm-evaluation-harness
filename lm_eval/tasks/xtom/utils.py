"""Paper-faithful lm-eval adapter for XToM (ACL 2026; arXiv:2506.02461).

The released XToM repository contains only a ZIP archive, not its evaluator. This
module therefore reconstructs the published protocol from Tables 6, 8--10, and
12--18 of the paper while keeping ``benchmarks/XToM`` strictly read-only.

Architecture:

* one task per (source benchmark, direct/CoT), with all five languages in each;
* one generated response per published prompt (except Negotiation belief/desire,
  whose three preference slots are intentionally batched in one prompt);
* rich per-document payloads aggregated into the language-specific paper columns;
* invalid generations remain in every headline denominator and are also exposed
  through a clearly labelled ``*_invalid_rate`` diagnostic.

Important reconstructions and release discrepancies are documented in README.md.
"""

from __future__ import annotations

import functools
import json
import random
import re
import unicodedata
import zipfile
from pathlib import Path

import datasets


LANGUAGES = ("en", "zh", "de", "fr", "ja")
_CHOICE_SEED = 99


# ---------------------------------------------------------------------------
# Released-data access (read ZIP in place; never extract under benchmarks/).
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _zip_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "benchmarks" / "XToM" / "XToM_DataSet.zip"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "XToM_DataSet.zip not found under any parent's benchmarks/XToM directory"
    )


@functools.lru_cache(maxsize=None)
def _read_entry(name: str):
    with zipfile.ZipFile(_zip_path()) as archive:
        with archive.open(name) as stream:
            return json.load(stream)


_ENTRY_SUFFIX = {
    "xfantom": {"en": "en", "zh": "cn", "de": "de", "fr": "fr", "ja": "ja"},
    "xtomi": {"en": "en", "zh": "ch", "de": "de", "fr": "fr", "ja": "ja"},
    "xnegtom": {"en": "en", "zh": "ch", "de": "de", "fr": "fr", "ja": "ja"},
}


def _entry_name(source: str, language: str) -> str:
    prefix = {"xfantom": "Xfantom", "xtomi": "XToMi", "xnegtom": "Xnegtom"}[source]
    return f"{prefix}_{_ENTRY_SUFFIX[source][language]}.json"


def _interleave(by_language: dict[str, list[dict]]) -> list[dict]:
    """Round-robin languages so a small ``--limit`` does not silently become EN-only."""
    docs = []
    max_len = max(len(items) for items in by_language.values())
    for index in range(max_len):
        for language in LANGUAGES:
            items = by_language[language]
            if index < len(items):
                docs.append(items[index])
    return docs


def _doc(prompt: str, target: str, meta: dict) -> dict:
    return {
        "prompt": prompt,
        "target": target,
        "language": meta["language"],
        "family": meta["family"],
        "qid": meta["qid"],
        # Keep heterogeneous list/scalar golds out of Arrow's nested inference.
        "meta": json.dumps(meta, ensure_ascii=False, sort_keys=True),
    }


# ---------------------------------------------------------------------------
# XToMi -- Table 17 prompts; paper selection = *_tom (not *_no_tom) + reality.
# ---------------------------------------------------------------------------


_XTOMI_TEMPLATES = {
    "en": "{story}\n{question}\nChoose from the following:\n(a) {choice0}\n(b) {choice1}\nKeep your answer concise. Answer with a single word.",
    "zh": "{story}\n{question}\n请从以下选项中选择：\n(a) {choice0}\n(b) {choice1}\n请保持答案简洁。使用单个词回答。",
    "de": "{story}\n{question}\nWählen Sie aus den folgenden Möglichkeiten:\n(a) {choice0}\n(b) {choice1}\nHalten Sie Ihre Antwort kurz. Antworten Sie mit einem einzigen Wort.",
    "fr": "{story}\n{question}\nChoisissez parmi les options suivantes :\n(a) {choice0}\n(b) {choice1}\nGardez votre réponse concise. Répondez avec un seul mot.",
    "ja": "{story}\n{question}\n以下から選択してください：\n(a) {choice0}\n(b) {choice1}\n簡潔に答えてください。一語で回答してください。",
}


def _xtomi_family(key: str) -> str | None:
    if key == "reality":
        return "reality"
    if "_no_tom" in key or not key.endswith("_tom"):
        return None
    if key.startswith("first_order_"):
        return "first_order"
    if key.startswith("second_order_"):
        return "second_order"
    return None


@functools.lru_cache(maxsize=1)
def _build_xtomi_docs() -> tuple[dict, ...]:
    # Five translated rows have a gold string that is not literally one of their
    # localized choices (FR: 3, JA: 2). Across every other evaluable row, the local
    # gold position equals the aligned English position exactly, so use the English
    # (id, question-key) position as the canonical recovery for those five rows.
    english_gold = {}
    for row in _read_entry(_entry_name("xtomi", "en")):
        for key, qa in row["question_types"].items():
            if _xtomi_family(key) is not None:
                english_gold[(row["id"], key)] = qa["containers"].index(qa["answer"])

    by_language = {}
    for language in LANGUAGES:
        docs = []
        for row in _read_entry(_entry_name("xtomi", language)):
            for key, qa in row["question_types"].items():
                family = _xtomi_family(key)
                if family is None:
                    continue
                choices = list(qa["containers"])
                if qa["answer"] in choices:
                    gold = choices.index(qa["answer"])
                    canonical = english_gold[(row["id"], key)]
                    if gold != canonical:
                        raise ValueError(
                            f"XToMi localized option order drifted: lang={language} id={row['id']} "
                            f"key={key} local={gold} english={canonical}"
                        )
                else:
                    gold = english_gold[(row["id"], key)]
                prompt = _XTOMI_TEMPLATES[language].format(
                    story=row["story"],
                    question=qa["question"],
                    choice0=choices[0],
                    choice1=choices[1],
                )
                meta = {
                    "source": "xtomi",
                    "language": language,
                    "family": family,
                    "qid": f"xtomi:{language}:{row['id']}:{key}",
                    "gold": gold,
                    "choices": choices,
                }
                # Log the canonical displayed target, including for the five rows
                # whose localized ``answer`` string drifted away from both choices.
                docs.append(_doc(prompt, str(choices[gold]), meta))
        by_language[language] = docs
    return tuple(_interleave(by_language))


def load_xtomi(**kwargs):
    return {"train": datasets.Dataset.from_list(list(_build_xtomi_docs()))}


# ---------------------------------------------------------------------------
# XFANToM -- Table 18 prompts; short context; fact + inaccessible belief only.
# ---------------------------------------------------------------------------


_XFANTOM_TEMPLATES = {
    "en": "{context}\nQuestion:\n{question}\n(a) {choice0}\n(b) {choice1}\nChoose an answer from above:",
    "zh": "{context}\n问题：\n{question}\n(a) {choice0}\n(b) {choice1}\n从上面选择一个答案：",
    "de": "{context}\nFrage:\n{question}\n(a) {choice0}\n(b) {choice1}\nWähle eine Antwort von oben:",
    "fr": "{context}\nQuestion :\n{question}\n(a) {choice0}\n(b) {choice1}\nChoisissez une réponse ci-dessus :",
    "ja": "{context}\n質問:\n{question}\n(a) {choice0}\n(b) {choice1}\n上記の中から答えを選んでください:",
}


def _binary_choices(qa: dict, rng: random.Random) -> tuple[list[str], int]:
    """Reuse FANToM's seed-99 wrong/correct binary shuffle (documented reconstruction)."""
    wrong, correct = qa["wrong_answer"], qa["correct_answer"]
    if rng.choice([True, False]):
        return [wrong, correct], 1
    return [correct, wrong], 0


def _xfantom_family(qa: dict) -> str:
    tom_type = qa["tom_type"]
    if tom_type == "first-order":
        return "belief_first"
    if tom_type == "second-order:acyclic":
        return "belief_acyclic"
    if tom_type == "second-order:cyclic":
        return "belief_cyclic"
    raise ValueError(f"unexpected inaccessible XFANToM tom_type: {tom_type!r}")


@functools.lru_cache(maxsize=1)
def _build_xfantom_docs() -> tuple[dict, ...]:
    by_language = {}
    for language in LANGUAGES:
        rng = random.Random(_CHOICE_SEED)
        docs = []
        seen_set_ids = set()
        for row in _read_entry(_entry_name("xfantom", language)):
            set_id = str(row["set_id"])
            if set_id in seen_set_ids:
                # Chinese release contains a second translation of set_id 24-0-0.
                continue
            seen_set_ids.add(set_id)
            context = row["short_context"].strip()

            candidates = [("fact", row["factQA"], "fact")]
            candidates.extend(
                (f"belief:{index}", qa, _xfantom_family(qa))
                for index, qa in enumerate(row["beliefQAs"])
                if qa["question_type"] == "tom:belief:inaccessible"
            )
            for qkey, qa, family in candidates:
                choices, gold = _binary_choices(qa, rng)
                prompt = _XFANTOM_TEMPLATES[language].format(
                    context=context,
                    question=qa["question"],
                    choice0=choices[0],
                    choice1=choices[1],
                )
                meta = {
                    "source": "xfantom",
                    "language": language,
                    "family": family,
                    "qid": f"xfantom:{language}:{set_id}:{qkey}",
                    "gold": gold,
                    "choices": choices,
                }
                docs.append(_doc(prompt, "(a)" if gold == 0 else "(b)", meta))
        by_language[language] = docs
    return tuple(_interleave(by_language))


def load_xfantom(**kwargs):
    return {"train": datasets.Dataset.from_list(list(_build_xfantom_docs()))}


# ---------------------------------------------------------------------------
# XNegotiationToM -- Tables 12--16 prompts and fixed A--D / A--I choices.
# ---------------------------------------------------------------------------


_NEG_CFG = {
    "en": {
        "pref_intro": (
            "Background: Here is a negotiation conversation for a camping trip. There are two agents who own some basic supplies and "
            "negotiate with each other to split the additional food packages, water bottles, and firewood to make their camping trip even better. "
            "Each of these items will be of either High, Medium or Low priority for these two agents. Each of the additional items only has an "
            "available quantity of 3. Please answer the following three questions using \"A\", \"B\", \"C\", \"D\" without any explanation."
        ),
        "intent_intro": (
            "Background: Here is a negotiation conversation for a camping trip. There are two agents who own some basic supplies and "
            "negotiate with each other to split the additional food packages, water bottles, and firewood to make their camping trip even better. "
            "Each of these items will be of either High, Medium or Low priority for these two agents. Each of the additional items only has an "
            "available quantity of 3."
        ),
        "history": "Dialogue History:",
        "people": ("Person 1", "Person 2"),
        "agents": ("Agent 1", "Agent 2"),
        "desire_agents": ("agent 1", "agent 2"),
        "belief_q": (
            "Question1: Based on the dialogue, what is the high preference for items {observer} thinks {other} is?",
            "Question2: Based on the dialogue, what is the medium preference for items {observer} thinks {other} is?",
            "Question3: Based on the dialogue, what is the low preference for items {observer} thinks {other} is?",
        ),
        "desire_q": (
            "Question1: What is {agent}'s high preference for items based on the dialogue history?",
            "Question2: What is {agent}'s medium preference for items based on the dialogue history?",
            "Question3: What is {agent}'s low preference for items based on the dialogue history?",
        ),
        "options": ("A.Not given", "B.Water", "C.Food", "D.Firewood"),
        "option_values": ("Not Given", "Water", "Food", "Firewood"),
        "answer": "Answer:",
        "intent_q": (
            "Question: What are the plausible intentions of {agent} expressed in '{utterance}'? Based on the dialogue history, select one "
            "or more strategies (i.e., 'A', 'B', 'C', ..., 'I') from the following choices and their definition. Please select 'A', 'B', 'C', ..., 'I' "
            "without any explanation."
        ),
        "intent_defs": (
            "A. Build-Rapport: Participants discussing topics apart from the negotiation, in an attempt to build a rapport with the partner.",
            "B. Show-Empathy: An utterance depicts empathy when there is evidence of positive acknowledgments or empathetic behavior towards a personal context of the partner.",
            "C. Promote-Coordination: Used when a participant promotes coordination among the two partners.",
            "D. Callout-Fairness: A callout to fairness for personal benefit, either when acknowledging a fair deal or when the opponent offers a deal that benefits them.",
            "E. Undermine-Requirements: Refers to the scenario where a participant undermines the requirements of their opponent.",
            "F. Discover-Preference: An attempt to discover the preference order of the opponent.",
            "G. Describe-Need: Refers to arguments for creating a personal need for an item in the negotiation.",
            "H. No-Need: When a participant points out that they do not need an item based on personal context.",
            "I. No-Intention: If no strategy is evident, the utterance is labeled as No-Intention.",
        ),
        "intent_values": (
            "Build-Rapport", "Show-Empathy", "Promote-Coordination", "Callout-Fairness",
            "Undermine-Requirements", "Discover-Preference", "Describe-Need", "No-Need", "No-Intention",
        ),
    },
    "zh": {
        "pref_intro": "背景：以下是一次关于露营旅行的谈判对话。两位参与者拥有一些基本物资，并相互协商如何分配额外的食物、水和火柴，以使他们的露营旅行更加愉快。这些物品对每位参与者的重要性优先级可以是高、中或低。每种额外物品的最大可用数量为3。请仅使用\"A\"、\"B\"、\"C\"、\"D\"作答，不需要解释。",
        "intent_intro": "背景：以下是一次关于露营旅行的谈判对话。两位参与者拥有一些基本物资，并相互协商如何分配额外的食物、水和火柴，以使他们的露营旅行更加愉快。这些物品对每位参与者的重要性优先级可以是高、中或低。每种额外物品的最大可用数量为3。",
        "history": "Dialogue History:",
        "people": ("人物1", "人物2"),
        "agents": ("人物1", "人物2"),
        "desire_agents": ("人物1", "人物2"),
        "belief_q": (
            "问题1：根据对话，{observer}认为{other}高优先级的物品是什么？",
            "问题2：根据对话，{observer}认为{other}中优先级的物品是什么？",
            "问题3：根据对话，{observer}认为{other}低优先级的物品是什么？",
        ),
        "desire_q": (
            "问题1：根据对话历史，{agent}高优先级的物品是什么？",
            "问题2：根据对话历史，{agent}中优先级的物品是什么？",
            "问题3：根据对话历史，{agent}低优先级的物品是什么？",
        ),
        "options": ("A.未提供", "B.水", "C.食物", "D.火柴"),
        "option_values": ("未提供", "水", "食物", "火柴"),
        "answer": "答案：",
        "intent_q": "问题：{agent}在\"{utterance}\"中表达的可能意图是什么？基于对话历史，从以下选项（即\"A\"、\"B\"、\"C\"、...、\"I\"）及其定义中选择一个或多个策略。请仅选择\"A\"、\"B\"、\"C\"、...、\"I\"，无需解释。",
        "intent_defs": (
            "A. 建立融洽关系：参与者讨论与谈判无关的主题，试图与对方建立融洽关系。",
            "B. 表达同情：当对方提到个人背景时，表现出积极的认可或同情行为的语句。",
            "C. 促进协调：当参与者促进双方之间的协调时使用。",
            "D. 呼吁公平：为了个人利益而呼吁公平，包括承认公平交易或对方提出有利于自己的交易时。",
            "E. 破坏要求：指参与者破坏对方需求的情境。",
            "F. 发现偏好：试图发现对方偏好顺序的行为。",
            "G. 描述需求：为某一物品的个人需求提供论据。",
            "H. 没有需求：根据个人背景指出他们不需要某一物品。",
            "I. 没有意图：如果没有明显的策略，则将语句标记为\"没有意图\"。",
        ),
        "intent_values": ("建立融洽关系", "表达同情", "促进协调", "呼吁公平", "破坏要求", "发现偏好", "描述需求", "没有需求", "没有意图"),
    },
    "de": {
        "pref_intro": "Hintergrund: Hier ist ein Verhandlungsgespräch für einen Campingausflug. Es gibt zwei Agenten, die einige grundlegende Vorräte besitzen und miteinander verhandeln, um die zusätzlichen Lebensmittelpakete, Wasserflaschen und Brennholz aufzuteilen, um ihren Campingausflug noch besser zu machen. Jeder dieser Gegenstände hat für diese beiden Agenten entweder eine hohe, mittlere oder niedrige Priorität. Für jeden der zusätzlichen Gegenstände ist nur eine Menge von 3 verfügbar. Bitte beantworten Sie die folgenden drei Fragen mit \"A\", \"B\", \"C\", \"D\" ohne Begründung.",
        "intent_intro": "Hintergrund: Hier ist ein Verhandlungsgespräch für einen Campingausflug. Es gibt zwei Agenten, die einige grundlegende Vorräte besitzen und miteinander verhandeln, um die zusätzlichen Lebensmittelpakete, Wasserflaschen und Brennholz aufzuteilen, um ihren Campingausflug noch besser zu machen. Jeder dieser Gegenstände hat für diese beiden Agenten entweder eine hohe, mittlere oder niedrige Priorität. Für jeden der zusätzlichen Gegenstände ist nur eine Menge von 3 verfügbar.",
        "history": "Dialogverlauf:",
        "people": ("Person 1", "Person 2"),
        "agents": ("Person 1", "Person 2"),
        "desire_agents": ("Person 1", "Person 2"),
        "belief_q": (
            "Frage1: Nach dem Dialog, was ist die hohe Präferenz für Gegenstände von {other} laut {observer}?",
            "Frage2: Nach dem Dialog, was ist die mittlere Präferenz für Gegenstände von {other} laut {observer}?",
            "Frage3: Nach dem Dialog, was ist die niedrige Präferenz für Gegenstände von {other} laut {observer}?",
        ),
        "desire_q": (
            "Frage1: Was ist die hohe Präferenz für Gegenstände von {agent} nach dem Dialogverlauf?",
            "Frage2: Was ist die mittlere Präferenz für Gegenstände von {agent} nach dem Dialogverlauf?",
            "Frage3: Was ist die niedrige Präferenz für Gegenstände von {agent} nach dem Dialogverlauf?",
        ),
        "options": ("A. Nicht angegeben", "B. Wasser", "C. Essen", "D. Brennholz"),
        "option_values": ("Nicht angegeben", "Wasser", "Essen", "Brennholz"),
        "answer": "Antwort:",
        "intent_q": "Frage: Was sind die plausiblen Absichten von {agent}, die in \"{utterance}\" ausgedrückt werden? Wählen Sie nach dem Dialogverlauf eine oder mehrere Strategien (d. h. \"A\", \"B\", \"C\", ..., \"I\") aus den folgenden Optionen und deren Definition aus. Wählen Sie \"A\", \"B\", \"C\", ..., \"I\" ohne Begründung aus.",
        "intent_defs": (
            "A. Beziehung aufbauen: Teilnehmer diskutieren Themen abseits der Verhandlung, um ein Vertrauensverhältnis zum Partner aufzubauen.",
            "B. Empathie zeigen: Eine Äußerung zeigt Empathie, wenn es Anzeichen für positive Anerkennungen oder empathisches Verhalten gegenüber einem persönlichen Kontext des Partners gibt.",
            "C. Koordination fördern: Wird verwendet, wenn ein Teilnehmer die Koordination zwischen den beiden Partnern fördert.",
            "D. Fairness einfordern: Ein Aufruf zur Fairness für persönlichen Vorteil, entweder wenn ein fairer Deal anerkannt wird oder wenn der Gegner einen Deal anbietet, der ihm Vorteile bringt.",
            "E. Anforderungen untergraben: Bezieht sich auf das Szenario, in dem ein Teilnehmer die Anforderungen seines Gegners untergräbt.",
            "F. Präferenz herausfinden: Ein Versuch, die Präferenzreihenfolge des Gegners herauszufinden.",
            "G. Bedarf beschreiben: Bezieht sich auf Argumente für die Schaffung eines persönlichen Bedarfs für einen Gegenstand in der Verhandlung.",
            "H. Kein Bedarf: Wenn ein Teilnehmer darauf hinweist, dass er einen Gegenstand aufgrund des persönlichen Kontexts nicht benötigt.",
            "I. Keine Absicht: Wenn keine Strategie erkennbar ist, wird die Äußerung als Keine Absicht gekennzeichnet.",
        ),
        "intent_values": ("Beziehung aufbauen", "Empathie zeigen", "Koordination fördern", "Fairness einfordern", "Anforderungen untergraben", "Präferenz herausfinden", "Bedarf beschreiben", "Kein Bedarf", "Keine Absicht"),
    },
    "fr": {
        "pref_intro": "Contexte : Voici une conversation de négociation pour un voyage de camping. Il y a deux agents qui possèdent quelques fournitures de base et négocient entre eux pour répartir les paquets de nourriture supplémentaires, les bouteilles d'eau et les bois de chauffage afin d'améliorer leur voyage de camping. Chacun de ces éléments sera d'une priorité Haute, Moyenne ou Faible pour ces deux agents. Chacun des articles supplémentaires n'a qu'une quantité disponible de 3. Veuillez répondre aux trois questions suivantes en utilisant \"A\", \"B\", \"C\", \"D\" sans aucune explication.",
        "intent_intro": "Contexte : Voici une conversation de négociation pour un voyage de camping. Il y a deux agents qui possèdent quelques fournitures de base et négocient entre eux pour répartir les paquets de nourriture supplémentaires, les bouteilles d'eau et les bois de chauffage afin d'améliorer leur voyage de camping. Chacun de ces éléments sera d'une priorité Haute, Moyenne ou Faible pour ces deux agents. Chacun des articles supplémentaires n'a qu'une quantité disponible de 3.",
        "history": "Historique de la conversation:",
        "people": ("Personne 1", "Personne 2"),
        "agents": ("Personne 1", "Personne 2"),
        "desire_agents": ("Personne 1", "Personne 2"),
        "belief_q": (
            "Question 1: D'après le dialogue, quels sont les articles que {observer} considère comme étant de haute priorité pour la {other}?",
            "Question 2: D'après le dialogue, quels sont les articles que {observer} considère comme étant de priorité moyenne pour la {other}?",
            "Question 3: D'après le dialogue, quels sont les articles que {observer} considère comme étant de faible priorité pour la {other}?",
        ),
        "desire_q": (
            "Question 1: Quels sont les articles de haute priorité pour la {agent}?",
            "Question 2: Quels sont les articles de priorité moyenne pour la {agent}?",
            "Question 3: Quels sont les articles de priorité faible pour la {agent}?",
        ),
        "options": ("A. Pas donné", "B. Eau", "C. Nourriture", "D. Bois de chauffage"),
        "option_values": ("Pas donné", "Eau", "Nourriture", "Bois de chauffage"),
        "answer": "Réponse:",
        "intent_q": "Question: Quelles sont les intentions plausibles de {agent} exprimées dans \"{utterance}\"? Sur la base de l'historique de la conversation, sélectionnez une ou plusieurs stratégies (c.-à-d., \"A\", \"B\", \"C\", ..., \"I\") parmi les choix suivants et leur définition. Veuillez sélectionner \"A\", \"B\", \"C\", ..., \"I\" sans aucune explication.",
        "intent_defs": (
            "A. Établir des relations: Les participants discutent de sujets autres que la négociation, dans le but de créer une relation avec le partenaire.",
            "B. Faire preuve d'empathie: Une énonciation montre de l'empathie lorsqu'il y a des preuves de reconnaissance positive ou de comportement empathique envers un contexte personnel du partenaire.",
            "C. Promouvoir la coordination: Utilisé lorsqu'un participant favorise la coordination entre les deux partenaires.",
            "D. Revendiquer l'équité: Un appel à l'équité pour un avantage personnel, soit en reconnaissant un accord équitable, soit lorsque l'adversaire propose un accord qui lui profite.",
            "E. Saper les exigences: Désigne le cas où un participant sape les exigences de son adversaire.",
            "F. Découvrir la préférence: Une tentative de découvrir l'ordre de préférence de l'adversaire.",
            "G. Décrire le besoin: Arguments visant à créer un besoin personnel pour un article dans la négociation.",
            "H. Aucun besoin: Lorsqu'un participant indique qu'il n'a pas besoin d'un article selon son contexte personnel.",
            "I. Aucune intention: Si aucune stratégie n'est évidente, l'énoncé est étiqueté comme Aucune-Intention.",
        ),
        "intent_values": ("établir des relations", "Faire preuve d'empathie", "Promouvoir la coordination", "Revendiquer l'équité", "Saper les exigences", "Découvrir la préférence", "Décrire le besoin", "Aucun besoin", "Aucune intention"),
    },
    "ja": {
        "pref_intro": "背景：以下はキャンプ旅行に関する交渉の会話です。二人の登場人物が基本的な用品を所有しており、追加の食べ物、水、薪を分配して、キャンプ旅行をより良くするために交渉しています。これらの各アイテムは、二人にとって「高」、「中」、「低」のいずれかの優先度を持ちます。追加アイテムは各々最大3個までしか利用できません。\n以下の3つの質問に\"A\"、\"B\"、\"C\"、\"D\"を使って説明なしで答えてください。",
        "intent_intro": "背景：以下はキャンプ旅行に関する交渉の会話です。二人の登場人物が基本的な用品を所有しており、追加の食べ物、水、薪を分配して、キャンプ旅行をより良くするために交渉しています。これらの各アイテムは、二人にとって「高」、「中」、「低」のいずれかの優先度を持ちます。追加アイテムは各々最大3個までしか利用できません。",
        "history": "対話履歴:",
        "people": ("人物1", "人物2"),
        "agents": ("人物1", "人物2"),
        "desire_agents": ("人物1", "人物2"),
        "belief_q": (
            "質問1: 対話に基づき、{observer}が{other}について考える「高」優先度のアイテムは何ですか？",
            "質問2: 対話に基づき、{observer}が{other}について考える「中」優先度のアイテムは何ですか？",
            "質問3: 対話に基づき、{observer}が{other}について考える「低」優先度のアイテムは何ですか？",
        ),
        "desire_q": (
            "質問1: 対話履歴に基づいて、{agent}の「高」優先度のアイテムは何ですか？",
            "質問2: 対話履歴に基づいて、{agent}の「中」優先度のアイテムは何ですか？",
            "質問3: 対話履歴に基づいて、{agent}の「低」優先度のアイテムは何ですか？",
        ),
        "options": ("A. 未提供", "B. 水", "C. 食べ物", "D. 薪"),
        "option_values": ("未提供", "水", "食べ物", "薪"),
        "answer": "回答:",
        "intent_q": "質問: {agent}が発した「{utterance}」において、考えられる意図は何ですか？対話履歴に基づいて、以下の選択肢から1つ以上の戦略（\"A\"、\"B\"、\"C\"、...、\"I\"）を選択してください。説明なしで\"A\"、\"B\"、\"C\"、...、\"I\"を選択してください。",
        "intent_defs": (
            "A. 信頼関係を築く：交渉とは別の話題を議論し、相手との信頼関係を築こうとする発言。",
            "B. 共感を示す：相手の個人的な文脈に対して、肯定的な応答や共感的な行動が見られる発言。",
            "C. 協調を促進する：両者間の協調を促進しようとする発言。",
            "D. 公平性を求める：自分に有利な条件を認める、または相手の提案が自分に利益をもたらすことを指摘する発言。",
            "E. 要件を損なう：相手の要件を軽視または否定する発言。",
            "F. 好みを見つける：相手の優先順位を探ろうとする発言。",
            "G. 需要を説明する：自分のアイテム需要を論じる発言。",
            "H. 要求なし：個人的な文脈に基づき、アイテムが必要ないことを指摘する発言。",
            "I. 意図なし：特定の戦略が明らかでない場合、このラベルが適用される。",
        ),
        "intent_values": ("信頼関係を築く", "共感を示す", "協調を促進する", "公平性を求める", "要件を損なう", "好みを見つける", "需要を説明する", "要求なし", "意図なし"),
    },
}


def _norm(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().replace("_", " ")
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _preference_prompt(language: str, dialogue: list[str], dimension: str, agent_index: int) -> str:
    cfg = _NEG_CFG[language]
    other_index = 1 - agent_index
    if dimension == "belief":
        questions = [
            q.format(observer=cfg["agents"][agent_index], other=cfg["agents"][other_index])
            for q in cfg["belief_q"]
        ]
    else:
        questions = [
            q.format(agent=cfg["desire_agents"][agent_index]) for q in cfg["desire_q"]
        ]
    blocks = []
    for question in questions:
        blocks.append("\n".join((question, *cfg["options"])))
    return "\n".join((cfg["pref_intro"], cfg["history"], *dialogue, *blocks, cfg["answer"]))


def _split_turn(turn: str) -> tuple[str, str]:
    for separator in (":", "："):
        if separator in turn:
            speaker, utterance = turn.split(separator, 1)
            return speaker.strip(), utterance.strip()
    return "", turn.strip()


def _intention_prompt(language: str, dialogue: list[str], agent: str, utterance: str) -> str:
    cfg = _NEG_CFG[language]
    question = cfg["intent_q"].format(agent=agent, utterance=utterance)
    return "\n".join(
        (cfg["intent_intro"], cfg["history"], *dialogue, question, *cfg["intent_defs"], cfg["answer"])
    )


def _slot_gold(row: dict, language: str, dimension: str, agent_number: int) -> list[int] | None:
    values = [row.get(f"agent{agent_number}_{dimension}_{level}") for level in ("high", "medium", "low")]
    if any(value is None or str(value) == "None" for value in values):
        return None
    index = {_norm(value): i for i, value in enumerate(_NEG_CFG[language]["option_values"])}
    try:
        return [index[_norm(value)] for value in values]
    except KeyError as exc:
        raise ValueError(
            f"unknown XNegotiationToM preference label: lang={language} dimension={dimension} values={values}"
        ) from exc


def _intent_gold(language: str, raw: str) -> list[int]:
    index = {_norm(value): i for i, value in enumerate(_NEG_CFG[language]["intent_values"])}
    labels = [piece.strip() for piece in str(raw).split(",") if piece.strip()]
    try:
        return sorted({index[_norm(label)] for label in labels})
    except KeyError as exc:
        raise ValueError(f"unknown XNegotiationToM intention label: lang={language} raw={raw!r}") from exc


@functools.lru_cache(maxsize=1)
def _build_xnegtom_docs() -> tuple[dict, ...]:
    by_language = {}
    for language in LANGUAGES:
        docs = []
        for row in _read_entry(_entry_name("xnegtom", language)):
            dialogue = list(row["dialogue"])
            dialogue_id = str(row["dialogue_id"])

            for agent_number in (1, 2):
                agent_index = agent_number - 1
                for dimension in ("belief", "desire"):
                    gold = _slot_gold(row, language, dimension, agent_number)
                    if gold is None:
                        continue
                    prompt = _preference_prompt(language, dialogue, dimension, agent_index)
                    target = ",".join(chr(ord("A") + value) for value in gold)
                    meta = {
                        "source": "xnegtom",
                        "language": language,
                        "family": dimension,
                        "qid": f"xnegtom:{language}:{dialogue_id}:{dimension}:agent{agent_number}",
                        "gold": gold,
                    }
                    docs.append(_doc(prompt, target, meta))

            # Released rows carry only intent/agent labels, not utterance text. The
            # valid intent slots align to the final N dialogue turns (N=1 or 2).
            valid_slots = [
                slot for slot in (1, 2)
                if row.get(f"utterance{slot}_agent") not in (None, "None")
                and row.get(f"utterance{slot}_intent") not in (None, "None")
            ]
            turns = dialogue[-len(valid_slots):] if valid_slots else []
            for slot, turn in zip(valid_slots, turns):
                stored_agent = str(row[f"utterance{slot}_agent"])
                speaker, utterance = _split_turn(turn)
                gold = _intent_gold(language, row[f"utterance{slot}_intent"])
                # One Chinese row (61-8, slot 1) mistranslates the stored agent as
                # 人物1 although the aligned English row and the utterance prefix are
                # Person/人物2. Prefer the speaker attached to the actual utterance.
                prompt_agent = speaker or stored_agent
                prompt = _intention_prompt(language, dialogue, prompt_agent, utterance)
                target = ",".join(chr(ord("A") + value) for value in gold)
                meta = {
                    "source": "xnegtom",
                    "language": language,
                    "family": "intention",
                    "qid": f"xnegtom:{language}:{dialogue_id}:intention:{slot}",
                    "gold": gold,
                }
                docs.append(_doc(prompt, target, meta))
        by_language[language] = docs
    return tuple(_interleave(by_language))


def load_xnegtom(**kwargs):
    return {"train": datasets.Dataset.from_list(list(_build_xnegtom_docs()))}


# ---------------------------------------------------------------------------
# Prompt variants and conservative reconstructed answer extraction.
# ---------------------------------------------------------------------------


def doc_to_text(doc):
    return doc["prompt"]


def doc_to_text_cot(doc):
    return doc["prompt"] + "\nlet's think step by step."


_CUE_RE = re.compile(
    r"(?is)(?:final\s+answer|the\s+answer\s+is|answer|antwort|réponse|答案|回答)\s*[:：]?"
)


def _answer_tail(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", " ", str(text)).strip()
    matches = list(_CUE_RE.finditer(text))
    if matches:
        text = text[matches[-1].end():].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def _extract_single(text: str, choices: list[str]) -> int | None:
    tail = _answer_tail(text)
    parenthesized = re.findall(r"(?i)\(([a-b])\)", tail)
    if parenthesized:
        return ord(parenthesized[-1].lower()) - ord("a")
    strong = re.findall(r"(?i)(?<![A-Za-z])([a-b])(?:[\s\.,:;)\]]|$)", tail)
    if strong:
        return ord(strong[-1].lower()) - ord("a")

    normalized_tail = _norm(tail)
    exact = [i for i, choice in enumerate(choices) if normalized_tail == _norm(choice)]
    if len(exact) == 1:
        return exact[0]
    contained = [i for i, choice in enumerate(choices) if _norm(choice) and _norm(choice) in normalized_tail]
    return contained[0] if len(contained) == 1 else None


def _extract_letters(text: str, upper: str, expected: int | None = None) -> list[int] | None:
    tail = _answer_tail(text)
    letters = re.findall(
        rf"(?i)(?<![A-Za-z])([a-{upper.lower()}])(?![A-Za-z])",
        tail,
    )
    if expected is not None and len(letters) != expected:
        return None
    if not letters:
        return None
    return [ord(letter.lower()) - ord("a") for letter in letters]


def _score_doc(doc, results: list[str]) -> dict:
    meta = json.loads(doc["meta"])
    raw = results[0] if results else ""
    family = meta["family"]
    if meta["source"] in ("xfantom", "xtomi"):
        prediction = _extract_single(raw, meta["choices"])
        valid = prediction is not None
        correct = valid and prediction == meta["gold"]
    elif family in ("belief", "desire"):
        prediction = _extract_letters(raw, "D", expected=3)
        valid = prediction is not None
        correct = valid and prediction == meta["gold"]
    elif family == "intention":
        extracted = _extract_letters(raw, "I")
        prediction = sorted(set(extracted)) if extracted is not None else []
        valid = extracted is not None
        correct = valid and prediction == meta["gold"]
    else:
        raise ValueError(f"unknown XToM family: {family!r}")
    return {
        "source": meta["source"],
        "language": meta["language"],
        "family": family,
        "qid": meta["qid"],
        "gold": meta["gold"],
        "prediction": prediction,
        "valid": bool(valid),
        "correct": bool(correct),
    }


# ---------------------------------------------------------------------------
# Metric registry and corpus aggregation (paper values are percentages).
# ---------------------------------------------------------------------------


_XFANTOM_COLUMNS = (
    "belief_acc", "belief_first_acc", "belief_second_acc",
    "belief_acyclic_acc", "belief_cyclic_acc", "fact_acc", "invalid_rate",
)
_XTOMI_COLUMNS = ("first_order_acc", "second_order_acc", "belief_acc", "reality_acc", "invalid_rate")
_XNEGTOM_COLUMNS = (
    "belief_exact_match", "desire_exact_match", "intention_micro_f1",
    "intention_macro_f1", "invalid_rate",
)

METRICS_XFANTOM = tuple(f"{language}_{column}" for language in LANGUAGES for column in _XFANTOM_COLUMNS)
METRICS_XTOMI = tuple(f"{language}_{column}" for language in LANGUAGES for column in _XTOMI_COLUMNS)
METRICS_XNEGTOM = tuple(f"{language}_{column}" for language in LANGUAGES for column in _XNEGTOM_COLUMNS)


def process_results_xfantom(doc, results):
    payload = _score_doc(doc, results)
    return {metric: payload for metric in METRICS_XFANTOM}


def process_results_xtomi(doc, results):
    payload = _score_doc(doc, results)
    return {metric: payload for metric in METRICS_XTOMI}


def process_results_xnegtom(doc, results):
    payload = _score_doc(doc, results)
    return {metric: payload for metric in METRICS_XNEGTOM}


def _mean_percent(items: list[dict]) -> float:
    return 100.0 * sum(bool(item["correct"]) for item in items) / len(items) if items else float("nan")


def _invalid_percent(items: list[dict]) -> float:
    return 100.0 * sum(not item["valid"] for item in items) / len(items) if items else float("nan")


def _intention_f1(items: list[dict], average: str) -> float:
    if not items:
        return float("nan")
    if average == "micro":
        tp = fp = fn = 0
        for item in items:
            gold, pred = set(item["gold"]), set(item["prediction"])
            tp += len(gold & pred)
            fp += len(pred - gold)
            fn += len(gold - pred)
        denominator = 2 * tp + fp + fn
        return 100.0 * (2 * tp / denominator if denominator else 0.0)

    scores = []
    for label in range(9):
        tp = fp = fn = 0
        for item in items:
            gold, pred = set(item["gold"]), set(item["prediction"])
            tp += label in gold and label in pred
            fp += label not in gold and label in pred
            fn += label in gold and label not in pred
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return 100.0 * sum(scores) / len(scores)


def _aggregate(metric: str, payloads) -> float:
    language, column = metric.split("_", 1)
    items = [item for item in payloads if item["language"] == language]
    if column == "invalid_rate":
        return _invalid_percent(items)

    if column == "belief_acc":
        if items and items[0]["source"] == "xfantom":
            selected = [item for item in items if item["family"].startswith("belief_")]
        else:
            selected = [item for item in items if item["family"] in ("first_order", "second_order")]
        return _mean_percent(selected)
    if column == "belief_first_acc":
        return _mean_percent([item for item in items if item["family"] == "belief_first"])
    if column == "belief_second_acc":
        return _mean_percent([item for item in items if item["family"] in ("belief_acyclic", "belief_cyclic")])
    if column == "belief_acyclic_acc":
        return _mean_percent([item for item in items if item["family"] == "belief_acyclic"])
    if column == "belief_cyclic_acc":
        return _mean_percent([item for item in items if item["family"] == "belief_cyclic"])
    if column == "fact_acc":
        return _mean_percent([item for item in items if item["family"] == "fact"])
    if column == "first_order_acc":
        return _mean_percent([item for item in items if item["family"] == "first_order"])
    if column == "second_order_acc":
        return _mean_percent([item for item in items if item["family"] == "second_order"])
    if column == "reality_acc":
        return _mean_percent([item for item in items if item["family"] == "reality"])
    if column == "belief_exact_match":
        return _mean_percent([item for item in items if item["family"] == "belief"])
    if column == "desire_exact_match":
        return _mean_percent([item for item in items if item["family"] == "desire"])
    if column == "intention_micro_f1":
        return _intention_f1([item for item in items if item["family"] == "intention"], "micro")
    if column == "intention_macro_f1":
        return _intention_f1([item for item in items if item["family"] == "intention"], "macro")
    raise KeyError(metric)


def _make_aggregation(metric: str):
    def aggregation(payloads):
        return _aggregate(metric, payloads)

    aggregation.__name__ = f"agg_{metric}"
    return aggregation


for _metric in set(METRICS_XFANTOM + METRICS_XTOMI + METRICS_XNEGTOM):
    globals()[f"agg_{_metric}"] = _make_aggregation(_metric)


def expected_counts() -> dict[str, dict[str, int]]:
    """Small model-free validation helper used by the README/test procedure."""
    out = {}
    for source, docs in (
        ("xfantom", _build_xfantom_docs()),
        ("xtomi", _build_xtomi_docs()),
        ("xnegtom", _build_xnegtom_docs()),
    ):
        counts = {language: 0 for language in LANGUAGES}
        for doc in docs:
            counts[doc["language"]] += 1
        out[source] = counts
    return out
