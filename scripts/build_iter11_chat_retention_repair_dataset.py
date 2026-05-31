#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


SYSTEM_TOOL = "Use available tools when the user's request requires the browser, terminal, or local files. Otherwise answer normally."
SYSTEM_CHAT = "You are a helpful assistant. Answer the user's question directly and completely. Do not use tools unless the user asks for current information, local files, or shell execution."


TOPICS = {
    "the moon landing": {
        "summary": "The first human Moon landing happened during NASA's Apollo 11 mission in July 1969.",
        "details": [
            "Neil Armstrong and Buzz Aldrin landed the lunar module Eagle in the Sea of Tranquility while Michael Collins orbited above in the command module Columbia.",
            "Armstrong stepped onto the surface first, followed by Aldrin, and they spent about two and a half hours outside collecting samples, taking photographs, and setting up experiments.",
            "The mission mattered scientifically because it returned lunar rocks and measurements, and culturally because it showed that a complex national engineering goal could be achieved under extreme constraints.",
        ],
    },
    "photosynthesis": {
        "summary": "Photosynthesis is the process plants, algae, and some bacteria use to turn light energy into chemical energy.",
        "details": [
            "Chlorophyll absorbs sunlight, and the organism uses that energy to convert carbon dioxide and water into sugars.",
            "Oxygen is released as a byproduct, which is why photosynthetic life is central to Earth's atmosphere and food webs.",
            "The process has light-dependent reactions that capture energy and carbon-fixation reactions that build carbohydrates.",
        ],
    },
    "how airplanes fly": {
        "summary": "Airplanes fly because their wings and engines work together to create lift, thrust, stability, and control.",
        "details": [
            "The engines push the airplane forward, causing air to move around the wings.",
            "The wing shape and angle of attack create a pressure difference and redirect air downward, producing lift.",
            "Pilots use control surfaces such as ailerons, elevators, and rudders to steer and keep the aircraft stable.",
        ],
    },
    "the Roman Empire": {
        "summary": "The Roman Empire was one of the most influential states in world history, centered on the Mediterranean world.",
        "details": [
            "It grew from the Roman Republic and reached its greatest territorial extent in the second century CE.",
            "Its roads, law, military organization, engineering, cities, and Latin language shaped later European and Mediterranean societies.",
            "The western empire fragmented in the fifth century, while the eastern empire continued for centuries as the Byzantine Empire.",
        ],
    },
    "black holes": {
        "summary": "A black hole is a region of spacetime where gravity is so strong that nothing inside the event horizon can escape.",
        "details": [
            "Most known black holes form when massive stars collapse after exhausting their nuclear fuel.",
            "They can be detected by their effects on nearby matter, such as hot accretion disks, stellar orbits, and gravitational waves from mergers.",
            "The event horizon is not a solid surface; it is the boundary beyond which escape would require traveling faster than light.",
        ],
    },
    "the water cycle": {
        "summary": "The water cycle describes how water moves between oceans, land, living things, and the atmosphere.",
        "details": [
            "Water evaporates from surfaces, condenses into clouds, falls as precipitation, and returns through runoff, groundwater, and transpiration.",
            "Solar energy and gravity drive the cycle, while temperature and geography shape local weather patterns.",
            "The cycle is essential for freshwater supply, ecosystems, agriculture, and climate regulation.",
        ],
    },
    "machine learning": {
        "summary": "Machine learning is a branch of computing where systems learn patterns from data instead of being programmed with every rule by hand.",
        "details": [
            "A model is trained on examples, adjusts internal parameters, and then uses the learned patterns to make predictions or generate outputs.",
            "Common approaches include supervised learning, unsupervised learning, reinforcement learning, and self-supervised learning.",
            "Good results depend on data quality, evaluation design, model capacity, and whether the training examples match the real task.",
        ],
    },
    "supply and demand": {
        "summary": "Supply and demand is a basic economic model for how prices and quantities are determined in a market.",
        "details": [
            "Demand describes how much buyers want at different prices, while supply describes how much sellers are willing to provide.",
            "When demand rises faster than supply, prices tend to increase; when supply rises faster than demand, prices tend to fall.",
            "Real markets also involve frictions such as regulation, market power, expectations, and production constraints.",
        ],
    },
}


EXPLANATION_PAIRS = [
    (
        "what is the difference between weather and climate?",
        "Weather is the short-term condition of the atmosphere, such as today's temperature, wind, rain, or humidity. Climate is the long-term pattern of weather in a region over many years. A cold day does not disprove a warming climate, because climate is measured through sustained averages and trends rather than a single event.",
    ),
    (
        "explain why batteries lose capacity over time",
        "Batteries lose capacity because their internal materials slowly change with use, age, heat, and charging cycles. In lithium-ion batteries, chemical side reactions consume active lithium, resistance increases, and electrode structures can degrade. The result is that the battery stores less energy and voltage drops sooner under load.",
    ),
    (
        "how does a compiler differ from an interpreter?",
        "A compiler translates source code into another form, often machine code or bytecode, before the program runs. An interpreter executes code more directly while the program is running. In practice many modern language runtimes mix both ideas, using interpretation for flexibility and compilation for speed.",
    ),
    (
        "why do seasons happen?",
        "Seasons happen because Earth's axis is tilted relative to its orbit around the Sun. During part of the year, one hemisphere receives more direct sunlight and longer days, producing warmer conditions. Six months later that hemisphere receives less direct sunlight and shorter days, producing cooler conditions.",
    ),
    (
        "what makes a good scientific experiment?",
        "A good scientific experiment tests a clear question with controlled conditions, measurable outcomes, and a method that others can repeat. It should isolate the variable being tested as much as possible and include enough observations to reduce random noise. The strongest experiments also define success criteria before looking at the results.",
    ),
]


SHORT_PAIRS = [
    ("hi there", "Hello. What would you like to work on?"),
    ("are you there", "Yes, I am here. What do you want to try next?"),
    ("thanks", "You're welcome."),
    ("what can you help with?", "I can help answer questions, reason through plans, write and review code, work with local files when tools are available, and help troubleshoot technical problems."),
    ("say done and nothing else", "done"),
]


def stable_split(case_id: str) -> str:
    value = int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "valid"
    return "test"


def read_tools(iter10_data: Path) -> list[dict[str, Any]]:
    first = json.loads((iter10_data / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    return first["tools"]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normal_rows(tools: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variants = [
        "tell me about {topic}",
        "explain {topic}",
        "give me a useful overview of {topic}",
        "what should I know about {topic}?",
        "can you describe {topic} in plain English?",
    ]
    for topic, info in TOPICS.items():
        answer = info["summary"] + " " + " ".join(info["details"])
        for idx, template in enumerate(variants):
            user = template.format(topic=topic)
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_TOOL if idx % 2 == 0 else SYSTEM_CHAT},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": answer},
                    ],
                    "tools": tools if idx % 2 == 0 else [],
                    "case_id": f"retention_topic_{topic.replace(' ', '_')}_{idx}",
                    "category": "normal_chat_long",
                    "kind": "no_tool_retention",
                    "source": "iter11:normal_topic",
                }
            )
    for idx, (user, answer) in enumerate(EXPLANATION_PAIRS):
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_TOOL},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": answer},
                ],
                "tools": tools,
                "case_id": f"retention_explain_{idx}",
                "category": "normal_chat_explain",
                "kind": "no_tool_retention",
                "source": "iter11:normal_explanation",
            }
        )
    for idx, (user, answer) in enumerate(SHORT_PAIRS):
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_CHAT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": answer},
                ],
                "tools": [],
                "case_id": f"retention_short_{idx}",
                "category": "normal_chat_short",
                "kind": "no_tool_retention",
                "source": "iter11:short_chat",
            }
        )
    # Oversample retention rows because the previous release overfit the model
    # toward short tool-router completions.
    expanded: list[dict[str, Any]] = []
    for row in rows:
        repeats = 16 if row["category"] != "normal_chat_short" else 6
        for i in range(repeats):
            clone = json.loads(json.dumps(row))
            clone["case_id"] = f"{row['case_id']}_rep{i:02d}"
            expanded.append(clone)
    rng.shuffle(expanded)
    return expanded


def load_iter10_rows(iter10_data: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        for line in (iter10_data / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row["case_id"] = f"iter10_{split}_{row.get('case_id', hashlib.sha1(line.encode()).hexdigest()[:12])}"
            row["source"] = row.get("source", "iter10")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter10-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=117)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tools = read_tools(args.iter10_data)
    rows = load_iter10_rows(args.iter10_data) + normal_rows(tools, rng)
    splits = {"train": [], "valid": [], "test": []}
    for row in rows:
        splits[stable_split(row["case_id"])].append(row)
    for split_rows in splits.values():
        rng.shuffle(split_rows)
    for split, split_rows in splits.items():
        write_jsonl(args.out / f"{split}.jsonl", split_rows)
    manifest = {
        "name": "iter11_chat_retention_repair",
        "seed": args.seed,
        "sources": [str(args.iter10_data), "synthetic normal-chat retention rows"],
        "splits": {k: len(v) for k, v in splits.items()},
        "category_counts": {
            category: sum(1 for split_rows in splits.values() for row in split_rows if row.get("category") == category)
            for category in sorted({row.get("category") for split_rows in splits.values() for row in split_rows})
        },
        "release_gate": "Normal chat must pass before any fused or GGUF artifact is republished.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
