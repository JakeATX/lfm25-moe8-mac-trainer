#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SYSTEM_CHAT = "You are a helpful assistant. Answer directly and completely."
SYSTEM_TOOL = "Use available tools when the user's request requires the browser, terminal, or local files. Otherwise answer normally."


TOPICS = [
    {
        "id": "compiler_interpreter",
        "prompts": [
            "what is the difference between a compiler and an interpreter?",
            "explain compilers vs interpreters in plain English",
            "how is a compiler different from an interpreter?",
            "give me a useful explanation of compiler versus interpreter",
        ],
        "answer": (
            "A compiler translates a program into another form, often machine code or bytecode, before the program is run. "
            "An interpreter reads and executes the program step by step while it is running. "
            "Compiled programs often start faster after compilation because much of the translation work has already happened. "
            "Interpreted programs can be easier to test interactively because the runtime can execute small pieces immediately. "
            "Many modern languages use a mix of both approaches, such as compiling to bytecode and then interpreting or just-in-time compiling that bytecode."
        ),
    },
    {
        "id": "black_holes",
        "prompts": [
            "what are black holes?",
            "explain black holes in plain English",
            "tell me what a black hole is",
            "give me a clear overview of black holes",
        ],
        "answer": (
            "A black hole is a region of space where gravity is so strong that nothing inside its event horizon can escape. "
            "Many black holes form when very massive stars collapse after running out of fuel. "
            "The event horizon is the boundary beyond which escape would require moving faster than light. "
            "Astronomers detect black holes by watching how they affect nearby stars, gas, light, and gravitational waves. "
            "They are not cosmic vacuum cleaners; objects far away orbit them according to gravity just as they would orbit another object with the same mass."
        ),
    },
    {
        "id": "weather_climate",
        "prompts": [
            "what is the difference between weather and climate?",
            "explain weather vs climate",
            "how are weather and climate different?",
            "what separates daily weather from climate?",
        ],
        "answer": (
            "Weather is the short-term condition of the atmosphere in a particular place, such as today's temperature, rain, wind, or humidity. "
            "Climate is the long-term pattern of weather for a region, usually measured over decades. "
            "A cold week does not disprove a warming climate, because climate is about averages and trends over long periods. "
            "Weather forecasts help with near-term planning, while climate data helps describe the typical range of conditions a place can expect. "
            "In short, weather is what is happening now, and climate is what usually happens over time."
        ),
    },
    {
        "id": "battery_capacity",
        "prompts": [
            "why do batteries lose capacity over time?",
            "why does my battery hold less charge as it ages?",
            "explain battery capacity loss",
            "what causes batteries to degrade?",
        ],
        "answer": (
            "Batteries lose capacity because their internal chemistry changes with use and age. "
            "In lithium-ion batteries, side reactions consume active lithium and change the electrolyte, so less chemical energy can move back and forth during charging. "
            "Heat, deep discharges, very high charge levels, and repeated charge cycles can accelerate that chemical wear. "
            "The electrodes can also develop structural damage, which raises resistance and reduces usable capacity. "
            "That is why an older battery may still work but hold noticeably less charge than it did when it was new."
        ),
    },
    {
        "id": "moon_landing",
        "prompts": [
            "tell me about the moon landing",
            "give me a normal explanation of the Apollo 11 moon landing",
            "what happened during the first moon landing?",
            "summarize the first human landing on the Moon",
        ],
        "answer": (
            "The first human Moon landing happened during NASA's Apollo 11 mission in July 1969. "
            "Neil Armstrong and Buzz Aldrin landed the lunar module Eagle in the Sea of Tranquility while Michael Collins orbited above in the command module Columbia. "
            "Armstrong became the first person to step onto the lunar surface, and Aldrin followed soon after. "
            "They collected samples, set up experiments, took photographs, and demonstrated that humans could work on another world. "
            "The mission became a defining achievement of space exploration and a major milestone in science, engineering, and international history."
        ),
    },
    {
        "id": "photosynthesis",
        "prompts": [
            "explain photosynthesis in plain English",
            "how does photosynthesis work?",
            "what is photosynthesis?",
            "give me a simple explanation of photosynthesis",
        ],
        "answer": (
            "Photosynthesis is how plants, algae, and some bacteria use light to make food. "
            "They take in carbon dioxide from the air and water from their surroundings. "
            "Using energy from sunlight, they turn those ingredients into sugars that store chemical energy. "
            "Oxygen is released as a byproduct, which is why photosynthesis is so important for life on Earth. "
            "The process gives plants the energy to grow and also helps maintain the oxygen in the atmosphere."
        ),
    },
]

NEGATIVES = [
    ("do not run anything; what command prints the current directory?", "The command is `pwd`. I do not need to run it to answer that."),
    ("write one sentence using the word browser", "The browser window was open, but the page had not finished loading."),
    ("do not use tools; explain what a terminal is", "A terminal is a text interface for giving commands to a computer and reading the output."),
    ("without searching, what does a file path mean?", "A file path is the address that tells the operating system where a file or folder is located."),
]


def row(case_id: str, system: str, prompt: str, answer: str, tools: list[dict] | None, category: str) -> dict:
    item = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "case_id": case_id,
        "category": category,
        "kind": "llamacpp_chat_retention",
        "source": "iter13:llamacpp_normal_chat_repair",
    }
    if tools is not None:
        item["tools"] = tools
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    tools = json.loads(args.tools_json.read_text(encoding="utf-8"))["tools"]
    rows = []
    for topic in TOPICS:
        for idx, prompt in enumerate(topic["prompts"]):
            for rep in range(8):
                system = SYSTEM_TOOL if rep % 2 == 0 else SYSTEM_CHAT
                rows.append(row(f"{topic['id']}_{idx:02d}_{rep:02d}", system, prompt, topic["answer"], tools if system == SYSTEM_TOOL else None, topic["id"]))
    for idx, (prompt, answer) in enumerate(NEGATIVES):
        for rep in range(10):
            system = SYSTEM_TOOL if rep % 2 == 0 else SYSTEM_CHAT
            rows.append(row(f"negative_{idx:02d}_{rep:02d}", system, prompt, answer, tools if system == SYSTEM_TOOL else None, "hard_negative_no_tool"))

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n = len(rows)
    train = rows[: int(n * 0.86)]
    valid = rows[int(n * 0.86): int(n * 0.93)]
    test = rows[int(n * 0.93):]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        with (args.out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for item in split:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": "iter13_llamacpp_chat_retention",
        "rows": {"train": len(train), "valid": len(valid), "test": len(test), "total": n},
        "purpose": "Repair early-stop normal chat behavior observed after GGUF conversion while preserving fixed Hermes tool routing.",
        "systems": [SYSTEM_CHAT, SYSTEM_TOOL],
        "topics": [t["id"] for t in TOPICS],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
